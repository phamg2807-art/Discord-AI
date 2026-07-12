const { Client, GatewayIntentBits, Partials, PermissionsBitField, ChannelType } = require('discord.js');
const { Mistral } = require('@mistralai/mistralai');
const http = require('http');
const fs = require('fs');
const path = require('path');

// ============================================================
// 1. Fake web server (keeps Render's port-binding check happy)
// ============================================================
const port = process.env.PORT || 3000;
http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('Bot is running securely!\n');
}).listen(port, () => {
    console.log(`Web server listening on port ${port} to satisfy Render requirements.`);
});

// ============================================================
// 2. Discord client
// ============================================================
const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent,
        GatewayIntentBits.GuildMembers,
        GatewayIntentBits.GuildModeration,
    ],
    partials: [Partials.Message, Partials.Channel],
});

const mistral = new Mistral({ apiKey: process.env.MISTRAL_API_KEY || '' });

// ============================================================
// 3. Config
// ============================================================
const TEXT_MODEL = 'mistral-large-latest';
const VISION_MODEL = 'pixtral-large-latest';
const HISTORY_LIMIT = 14;
const FACT_LIMIT = 20;
const LEARN_EVERY = 3;
const DATA_FILE = path.join(__dirname, 'memory.json');
const DEFAULT_PERSONA =
    'Bạn là một trợ lý AI thông minh, thân thiện, thích ứng theo từng máy chủ và từng người dùng. ' +
    'Trả lời ngắn gọn, tự nhiên, đúng trọng tâm. Khi người dùng gửi ảnh, hãy mô tả/phân tích ảnh đó. ' +
    'Khi người dùng gửi code, hãy đọc kỹ, giải thích rõ ràng và trả code trong khối markdown (```lang). ' +
    'Nếu người dùng là Admin và yêu cầu quản lý server (tạo/xoá/đổi tên kênh, quản lý vai trò, kick/timeout), ' +
    'hãy dùng công cụ (tool) tương ứng thay vì chỉ mô tả cách làm.';

// ============================================================
// 4. Persistent store
// ============================================================
let store = { guilds: {}, channels: {}, users: {} };

function loadStore() {
    try {
        if (fs.existsSync(DATA_FILE)) store = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
    } catch (e) {
        console.error('⚠️  Could not read memory.json, starting fresh:', e.message);
    }
}
loadStore();

let saveTimer = null;
function saveStoreSoon() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
        fs.writeFile(DATA_FILE, JSON.stringify(store, null, 2), (err) => {
            if (err) console.error('⚠️  Could not save memory.json:', err.message);
        });
    }, 1500);
}

function getGuild(guildId) {
    if (!store.guilds[guildId]) store.guilds[guildId] = { persona: DEFAULT_PERSONA };
    return store.guilds[guildId];
}
function getUser(userId, displayName) {
    if (!store.users[userId]) store.users[userId] = { name: displayName, facts: [], msgCount: 0, lastLearn: 0 };
    if (displayName) store.users[userId].name = displayName;
    return store.users[userId];
}
function getChannelHistory(channelId) {
    if (!store.channels[channelId]) store.channels[channelId] = [];
    return store.channels[channelId];
}
function pushHistory(channelId, entry) {
    const hist = getChannelHistory(channelId);
    hist.push(entry);
    while (hist.length > HISTORY_LIMIT) hist.shift();
}

// ============================================================
// 5. Learning
// ============================================================
async function learnAboutUser(userId, recentUserText) {
    try {
        const user = getUser(userId);
        const resp = await mistral.chat.complete({
            model: TEXT_MODEL,
            messages: [
                {
                    role: 'system',
                    content:
                        'Trích xuất tối đa 3 sự thật NGẮN GỌN, lâu dài và hữu ích về người dùng từ đoạn tin nhắn dưới đây. ' +
                        'CHỈ trả về JSON: {"facts": ["...", "..."]}. Nếu không có gì đáng nhớ, trả {"facts": []}.',
                },
                { role: 'user', content: recentUserText },
            ],
            responseFormat: { type: 'json_object' },
            maxTokens: 300,
        });
        const parsed = JSON.parse(resp.choices[0].message.content);
        if (Array.isArray(parsed.facts)) {
            for (const fact of parsed.facts) {
                if (fact && typeof fact === 'string' && !user.facts.includes(fact)) user.facts.push(fact);
            }
            while (user.facts.length > FACT_LIMIT) user.facts.shift();
            saveStoreSoon();
        }
    } catch (e) {
        console.error('Fact-learning skipped (non-fatal):', e.message);
    }
}

// ============================================================
// 6. Prompt / content helpers
// ============================================================
function buildSystemPrompt(guildId, userId, isAdmin) {
    const persona = getGuild(guildId).persona;
    const user = getUser(userId);
    let sys = persona;
    if (user.facts.length) {
        sys += `\n\nNhững điều bạn đã biết về người dùng này (dùng tự nhiên, đừng liệt kê máy móc):\n- ${user.facts.join('\n- ')}`;
    }
    sys += isAdmin
        ? '\n\nNgười đang nhắn tin là Admin của server này, được phép dùng mọi công cụ quản trị.'
        : '\n\nNgười đang nhắn tin KHÔNG phải Admin — nếu họ yêu cầu hành động quản trị, hãy giải thích rằng chỉ Admin mới có thể làm việc đó, đừng gọi tool.';
    return sys;
}
function looksLikeCode(text) {
    return /```/.test(text) || /\b(function|const|let|def |class |import |#include|SELECT |public static)\b/.test(text);
}
async function collectImageUrls(message) {
    const urls = [];
    for (const att of message.attachments.values()) {
        if (att.contentType && att.contentType.startsWith('image/')) urls.push(att.url);
    }
    return urls;
}

// ============================================================
// 7. Admin tool definitions (function calling schema for Mistral)
// ============================================================
const CHANNEL_TYPE_MAP = {
    text: ChannelType.GuildText,
    voice: ChannelType.GuildVoice,
    category: ChannelType.GuildCategory,
    announcement: ChannelType.GuildAnnouncement,
    forum: ChannelType.GuildForum,
};

const tools = [
    {
        type: 'function',
        function: {
            name: 'list_channels',
            description: 'List all channels in the server, grouped by category.',
            parameters: { type: 'object', properties: {} },
        },
    },
    {
        type: 'function',
        function: {
            name: 'find_user',
            description: 'Search server members by username or nickname (partial match).',
            parameters: {
                type: 'object',
                properties: { query: { type: 'string', description: 'Name or partial name to search for' } },
                required: ['query'],
            },
        },
    },
    {
        type: 'function',
        function: {
            name: 'create_channel',
            description: 'Create a new channel in the server.',
            parameters: {
                type: 'object',
                properties: {
                    name: { type: 'string' },
                    type: { type: 'string', enum: ['text', 'voice', 'category', 'announcement', 'forum'] },
                    category: { type: 'string', description: 'Name of an existing category to place it under (optional)' },
                    topic: { type: 'string', description: 'Optional channel topic/description' },
                },
                required: ['name', 'type'],
            },
        },
    },
    {
        type: 'function',
        function: {
            name: 'delete_channel',
            description: 'Delete an existing channel by name.',
            parameters: {
                type: 'object',
                properties: { channel_name: { type: 'string' } },
                required: ['channel_name'],
            },
        },
    },
    {
        type: 'function',
        function: {
            name: 'rename_channel',
            description: 'Rename an existing channel.',
            parameters: {
                type: 'object',
                properties: { old_name: { type: 'string' }, new_name: { type: 'string' } },
                required: ['old_name', 'new_name'],
            },
        },
    },
    {
        type: 'function',
        function: {
            name: 'set_channel_topic',
            description: 'Set/update the topic (description) of a text channel.',
            parameters: {
                type: 'object',
                properties: { channel_name: { type: 'string' }, topic: { type: 'string' } },
                required: ['channel_name', 'topic'],
            },
        },
    },
    {
        type: 'function',
        function: {
            name: 'create_role',
            description: 'Create a new role in the server.',
            parameters: {
                type: 'object',
                properties: {
                    name: { type: 'string' },
                    color: { type: 'string', description: 'Hex color like #FF5733 (optional)' },
                    mentionable: { type: 'boolean' },
                },
                required: ['name'],
            },
        },
    },
    {
        type: 'function',
        function: {
            name: 'assign_role',
            description: 'Assign an existing role to a member.',
            parameters: {
                type: 'object',
                properties: { user_query: { type: 'string' }, role_name: { type: 'string' } },
                required: ['user_query', 'role_name'],
            },
        },
    },
    {
        type: 'function',
        function: {
            name: 'remove_role',
            description: 'Remove a role from a member.',
            parameters: {
                type: 'object',
                properties: { user_query: { type: 'string' }, role_name: { type: 'string' } },
                required: ['user_query', 'role_name'],
            },
        },
    },
    {
        type: 'function',
        function: {
            name: 'kick_user',
            description: 'Kick a member from the server.',
            parameters: {
                type: 'object',
                properties: { user_query: { type: 'string' }, reason: { type: 'string' } },
                required: ['user_query'],
            },
        },
    },
    {
        type: 'function',
        function: {
            name: 'timeout_user',
            description: 'Timeout (mute) a member for a duration.',
            parameters: {
                type: 'object',
                properties: {
                    user_query: { type: 'string' },
                    minutes: { type: 'number', description: 'Timeout duration in minutes (max 40320 = 28 days)' },
                    reason: { type: 'string' },
                },
                required: ['user_query', 'minutes'],
            },
        },
    },
    {
        type: 'function',
        function: {
            name: 'ban_user',
            description: 'Ban a member from the server.',
            parameters: {
                type: 'object',
                properties: { user_query: { type: 'string' }, reason: { type: 'string' } },
                required: ['user_query'],
            },
        },
    },
];

// Actions that change/remove server state — require Administrator (checked again here as defense-in-depth)
const DESTRUCTIVE = new Set([
    'create_channel', 'delete_channel', 'rename_channel', 'set_channel_topic',
    'create_role', 'assign_role', 'remove_role', 'kick_user', 'timeout_user', 'ban_user',
]);

function findMember(guild, query) {
    const q = query.toLowerCase().replace(/^@/, '');
    return guild.members.cache.find(
        (m) =>
            m.user.username.toLowerCase().includes(q) ||
            (m.nickname && m.nickname.toLowerCase().includes(q)) ||
            m.displayName.toLowerCase().includes(q)
    );
}
function findRole(guild, name) {
    const q = name.toLowerCase();
    return guild.roles.cache.find((r) => r.name.toLowerCase() === q || r.name.toLowerCase().includes(q));
}
function findChannel(guild, name) {
    const q = name.toLowerCase().replace(/^#/, '');
    return guild.channels.cache.find((c) => c.name.toLowerCase() === q || c.name.toLowerCase().includes(q));
}

// ============================================================
// 8. Tool execution — the actual Discord.js side effects
// ============================================================
async function executeTool(guild, name, args) {
    switch (name) {
        case 'list_channels': {
            const cats = guild.channels.cache.filter((c) => c.type === ChannelType.GuildCategory);
            const lines = [];
            for (const cat of cats.values()) {
                const children = guild.channels.cache.filter((c) => c.parentId === cat.id);
                lines.push(`📁 ${cat.name}: ${children.map((c) => c.name).join(', ') || '(trống)'}`);
            }
            const uncategorized = guild.channels.cache.filter((c) => !c.parentId && c.type !== ChannelType.GuildCategory);
            if (uncategorized.size) lines.push(`(không danh mục): ${uncategorized.map((c) => c.name).join(', ')}`);
            return { ok: true, result: lines.join('\n') || 'Server chưa có kênh nào.' };
        }
        case 'find_user': {
            const member = findMember(guild, args.query);
            if (!member) return { ok: false, result: `Không tìm thấy người dùng khớp với "${args.query}".` };
            return {
                ok: true,
                result: `${member.user.tag} (nickname: ${member.nickname || 'không có'}), vai trò: ${member.roles.cache.map((r) => r.name).join(', ')}`,
            };
        }
        case 'create_channel': {
            const opts = { name: args.name, type: CHANNEL_TYPE_MAP[args.type] ?? ChannelType.GuildText };
            if (args.topic) opts.topic = args.topic;
            if (args.category) {
                const cat = findChannel(guild, args.category);
                if (cat && cat.type === ChannelType.GuildCategory) opts.parent = cat.id;
            }
            const created = await guild.channels.create(opts);
            return { ok: true, result: `Đã tạo kênh #${created.name}.` };
        }
        case 'delete_channel': {
            const ch = findChannel(guild, args.channel_name);
            if (!ch) return { ok: false, result: `Không tìm thấy kênh "${args.channel_name}".` };
            const chName = ch.name;
            await ch.delete();
            return { ok: true, result: `Đã xoá kênh #${chName}.` };
        }
        case 'rename_channel': {
            const ch = findChannel(guild, args.old_name);
            if (!ch) return { ok: false, result: `Không tìm thấy kênh "${args.old_name}".` };
            await ch.setName(args.new_name);
            return { ok: true, result: `Đã đổi tên kênh thành #${args.new_name}.` };
        }
        case 'set_channel_topic': {
            const ch = findChannel(guild, args.channel_name);
            if (!ch || !('setTopic' in ch)) return { ok: false, result: `Không tìm thấy kênh văn bản "${args.channel_name}".` };
            await ch.setTopic(args.topic);
            return { ok: true, result: `Đã cập nhật chủ đề cho #${ch.name}.` };
        }
        case 'create_role': {
            const role = await guild.roles.create({
                name: args.name,
                color: args.color || undefined,
                mentionable: !!args.mentionable,
            });
            return { ok: true, result: `Đã tạo vai trò "${role.name}".` };
        }
        case 'assign_role': {
            const member = findMember(guild, args.user_query);
            const role = findRole(guild, args.role_name);
            if (!member) return { ok: false, result: `Không tìm thấy người dùng "${args.user_query}".` };
            if (!role) return { ok: false, result: `Không tìm thấy vai trò "${args.role_name}".` };
            await member.roles.add(role);
            return { ok: true, result: `Đã gán vai trò "${role.name}" cho ${member.user.tag}.` };
        }
        case 'remove_role': {
            const member = findMember(guild, args.user_query);
            const role = findRole(guild, args.role_name);
            if (!member) return { ok: false, result: `Không tìm thấy người dùng "${args.user_query}".` };
            if (!role) return { ok: false, result: `Không tìm thấy vai trò "${args.role_name}".` };
            await member.roles.remove(role);
            return { ok: true, result: `Đã gỡ vai trò "${role.name}" khỏi ${member.user.tag}.` };
        }
        case 'kick_user': {
            const member = findMember(guild, args.user_query);
            if (!member) return { ok: false, result: `Không tìm thấy người dùng "${args.user_query}".` };
            if (!member.kickable) return { ok: false, result: `Tôi không có quyền kick ${member.user.tag} (vai trò của họ cao hơn bot).` };
            const tag = member.user.tag;
            await member.kick(args.reason || 'Không có lý do cụ thể');
            return { ok: true, result: `Đã kick ${tag}.` };
        }
        case 'timeout_user': {
            const member = findMember(guild, args.user_query);
            if (!member) return { ok: false, result: `Không tìm thấy người dùng "${args.user_query}".` };
            const ms = Math.min(Math.max(args.minutes, 1), 40320) * 60 * 1000;
            await member.timeout(ms, args.reason || 'Không có lý do cụ thể');
            return { ok: true, result: `Đã timeout ${member.user.tag} trong ${args.minutes} phút.` };
        }
        case 'ban_user': {
            const member = findMember(guild, args.user_query);
            if (!member) return { ok: false, result: `Không tìm thấy người dùng "${args.user_query}".` };
            if (!member.bannable) return { ok: false, result: `Tôi không có quyền ban ${member.user.tag}.` };
            const tag = member.user.tag;
            await member.ban({ reason: args.reason || 'Không có lý do cụ thể' });
            return { ok: true, result: `Đã ban ${tag}.` };
        }
        default:
            return { ok: false, result: `Không rõ công cụ "${name}".` };
    }
}

// ============================================================
// 9. Text commands (persona / memory management)
// ============================================================
async function handleCommand(message, cleanPrompt) {
    const lower = cleanPrompt.toLowerCase();

    if (lower.startsWith('!persona ')) {
        const newPersona = cleanPrompt.slice('!persona '.length).trim();
        if (!newPersona) return message.reply('Hãy nhập mô tả persona sau lệnh, ví dụ: `!persona Bạn là một hải tặc dí dỏm`.');
        getGuild(message.guildId).persona = newPersona;
        saveStoreSoon();
        return message.reply(`✅ Đã đổi giọng/persona của tôi cho server này thành:\n> ${newPersona}`);
    }
    if (lower === '!resetpersona') {
        getGuild(message.guildId).persona = DEFAULT_PERSONA;
        saveStoreSoon();
        return message.reply('✅ Đã đặt lại persona mặc định cho server này.');
    }
    if (lower === '!forgetme') {
        delete store.users[message.author.id];
        saveStoreSoon();
        return message.reply('🧹 Đã xoá toàn bộ thông tin tôi học được về bạn.');
    }
    if (lower === '!whatyouknow') {
        const user = getUser(message.author.id);
        if (!user.facts.length) return message.reply('Tôi chưa học được điều gì đặc biệt về bạn cả!');
        return message.reply(`Đây là những gì tôi nhớ về bạn:\n- ${user.facts.join('\n- ')}`);
    }
    return null;
}

// ============================================================
// 10. Main handler
// ============================================================
client.once('ready', () => {
    console.log(`Bot logged in successfully as: ${client.user.tag} 🚀`);
});

client.on('messageCreate', async (message) => {
    if (message.author.bot) return;
    if (!message.mentions.has(client.user)) return;

    const cleanPrompt = message.content.replace(/<@!?\d+>/g, '').trim();
    if (!cleanPrompt && message.attachments.size === 0) {
        return message.reply('Xin chào! Tôi có thể giúp gì cho bạn hôm nay?');
    }

    if (cleanPrompt.startsWith('!')) {
        const handled = await handleCommand(message, cleanPrompt);
        if (handled !== null) return;
    }

    try {
        await message.channel.sendTyping();

        const isAdmin = message.member?.permissions.has(PermissionsBitField.Flags.Administrator) ?? false;
        const imageUrls = await collectImageUrls(message);
        const hasImages = imageUrls.length > 0;
        const user = getUser(message.author.id, message.author.username);
        const channelId = message.channelId;

        let userContent;
        if (hasImages) {
            userContent = [
                { type: 'text', text: cleanPrompt || 'Hãy mô tả và phân tích (các) hình ảnh này.' },
                // NOTE: the JS SDK requires camelCase "imageUrl", NOT "image_url" — using the
                // snake_case key causes a silent validation failure that surfaces as a generic error.
                ...imageUrls.map((url) => ({ type: 'image_url', imageUrl: url })),
            ];
        } else {
            userContent = cleanPrompt;
        }

        const systemPrompt = buildSystemPrompt(message.guildId, message.author.id, isAdmin);
        const history = getChannelHistory(channelId);
        let messages = [{ role: 'system', content: systemPrompt }, ...history, { role: 'user', content: userContent }];

        const model = hasImages ? VISION_MODEL : TEXT_MODEL;
        const isCode = looksLikeCode(cleanPrompt);

        // Only offer admin tools to admins — non-admins never even get the option
        const activeTools = isAdmin ? tools : undefined;

        let response = await mistral.chat.complete({
            model,
            messages,
            tools: activeTools,
            toolChoice: activeTools ? 'auto' : undefined,
            maxTokens: isCode ? 1800 : 900,
        });

        let choice = response.choices[0];

        // Tool-calling loop (handles the model chaining a couple of actions)
        let guard = 0;
        while (choice.message.toolCalls && choice.message.toolCalls.length && guard < 5) {
            guard++;
            messages.push(choice.message);
            for (const call of choice.message.toolCalls) {
                const fnName = call.function.name;
                let args = {};
                try { args = JSON.parse(call.function.arguments); } catch (_) {}

                let toolResult;
                if (!isAdmin && DESTRUCTIVE.has(fnName)) {
                    toolResult = { ok: false, result: 'Chỉ Admin mới được phép thực hiện hành động này.' };
                } else {
                    try {
                        toolResult = await executeTool(message.guild, fnName, args);
                    } catch (e) {
                        console.error(`Tool ${fnName} failed:`, e.message);
                        toolResult = { ok: false, result: `Lỗi khi thực hiện "${fnName}": ${e.message}` };
                    }
                }
                messages.push({
                    role: 'tool',
                    name: fnName,
                    toolCallId: call.id,
                    content: JSON.stringify(toolResult),
                });
            }
            response = await mistral.chat.complete({ model: TEXT_MODEL, messages, tools: activeTools, toolChoice: 'auto', maxTokens: 900 });
            choice = response.choices[0];
        }

        const botReply = choice.message.content || '(không có phản hồi)';

        pushHistory(channelId, { role: 'user', content: hasImages ? `${cleanPrompt} [đã gửi ${imageUrls.length} ảnh]` : cleanPrompt });
        pushHistory(channelId, { role: 'assistant', content: botReply });
        saveStoreSoon();

        user.msgCount += 1;
        if (user.msgCount - user.lastLearn >= LEARN_EVERY) {
            user.lastLearn = user.msgCount;
            learnAboutUser(message.author.id, cleanPrompt);
        }

        if (botReply.length <= 2000) {
            await message.reply(botReply);
        } else {
            const chunks = botReply.match(/[\s\S]{1,1990}/g) || [];
            for (const chunk of chunks) await message.channel.send(chunk);
        }
    } catch (error) {
        // Log full detail (SDK errors often carry a `.body` or `.rawValue` with the real reason)
        console.error('Execution Error:', error?.body || error?.rawValue || error);
        await message.reply('Đã có lỗi xảy ra khi kết nối tới Mistral AI. Hãy thử lại sau nhé!');
    }
});

process.on('SIGTERM', () => {
    fs.writeFileSync(DATA_FILE, JSON.stringify(store, null, 2));
    process.exit(0);
});

client.login(process.env.DISCORD_TOKEN);
