const { Client, GatewayIntentBits, Partials } = require('discord.js');
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
    ],
    partials: [Partials.Message, Partials.Channel],
});

const mistral = new Mistral({ apiKey: process.env.MISTRAL_API_KEY || '' });

// ============================================================
// 3. Config
// ============================================================
const TEXT_MODEL = 'mistral-large-latest';   // text-only requests
const VISION_MODEL = 'pixtral-large-latest'; // requests that include images
const HISTORY_LIMIT = 14;   // short-term messages kept per channel (context window)
const FACT_LIMIT = 20;      // long-term "learned" facts kept per user
const LEARN_EVERY = 3;      // run fact-extraction every N user messages
const DATA_FILE = path.join(__dirname, 'memory.json');
const DEFAULT_PERSONA =
    'Bạn là một trợ lý AI thông minh, thân thiện, thích ứng theo từng máy chủ và từng người dùng. ' +
    'Trả lời ngắn gọn, tự nhiên, đúng trọng tâm. Khi người dùng gửi ảnh, hãy mô tả/phân tích ảnh đó. ' +
    'Khi người dùng gửi code, hãy đọc kỹ, giải thích rõ ràng và trả code trong khối markdown (```lang).';

// ============================================================
// 4. Persistent store: guild personas, channel history, user profiles
//    NOTE: Render's default filesystem is ephemeral (wiped on redeploy/
//    restart). This JSON file survives while the instance is alive, but
//    for durability across deploys use a Render Persistent Disk, or swap
//    this for a real DB (SQLite/Postgres/Redis) later — the load/save
//    functions below are the only place you'd need to change.
// ============================================================
let store = {
    guilds: {},   // guildId   -> { persona: string }
    channels: {}, // channelId -> [{ role, content }]
    users: {},    // userId    -> { name, facts: [], msgCount, lastLearn }
};

function loadStore() {
    try {
        if (fs.existsSync(DATA_FILE)) {
            store = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
        }
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
    if (!store.users[userId]) {
        store.users[userId] = { name: displayName, facts: [], msgCount: 0, lastLearn: 0 };
    }
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
// 5. "Learning" — periodically distill durable facts about a user
//    from the recent conversation, using the model itself.
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
                        'Trích xuất tối đa 3 sự thật NGẮN GỌN, lâu dài và hữu ích về người dùng ' +
                        '(sở thích, nghề nghiệp/dự án đang làm, mục tiêu, phong cách giao tiếp ưa thích, v.v.) ' +
                        'từ đoạn tin nhắn dưới đây. Bỏ qua thông tin nhất thời/không quan trọng. ' +
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
                if (fact && typeof fact === 'string' && !user.facts.includes(fact)) {
                    user.facts.push(fact);
                }
            }
            while (user.facts.length > FACT_LIMIT) user.facts.shift();
            saveStoreSoon();
        }
    } catch (e) {
        console.error('Fact-learning skipped (non-fatal):', e.message);
    }
}

// ============================================================
// 6. Helpers: build the multimodal message, detect code/images
// ============================================================
function buildSystemPrompt(guildId, userId) {
    const persona = getGuild(guildId).persona;
    const user = getUser(userId);
    let sys = persona;
    if (user.facts.length) {
        sys += `\n\nNhững điều bạn đã biết về người dùng này (dùng tự nhiên, đừng liệt kê máy móc):\n- ${user.facts.join('\n- ')}`;
    }
    return sys;
}

function looksLikeCode(text) {
    return /```/.test(text) || /\b(function|const|let|def |class |import |#include|SELECT |public static)\b/.test(text);
}

async function collectImageUrls(message) {
    const urls = [];
    for (const att of message.attachments.values()) {
        if (att.contentType && att.contentType.startsWith('image/')) {
            urls.push(att.url);
        }
    }
    return urls;
}

// ============================================================
// 7. Slash-free "commands" (typed right after mentioning the bot)
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

    return null; // not a command
}

// ============================================================
// 8. Main handler
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

    // Server-scoped commands (persona control, memory control)
    if (cleanPrompt.startsWith('!')) {
        const handled = await handleCommand(message, cleanPrompt);
        if (handled !== null) return;
    }

    try {
        await message.channel.sendTyping();

        const imageUrls = await collectImageUrls(message);
        const hasImages = imageUrls.length > 0;
        const user = getUser(message.author.id, message.author.username);
        const channelId = message.channelId;

        // Build the user turn (multimodal if images are attached)
        let userContent;
        if (hasImages) {
            userContent = [
                { type: 'text', text: cleanPrompt || 'Hãy mô tả và phân tích (các) hình ảnh này.' },
                ...imageUrls.map((url) => ({ type: 'image_url', image_url: url })),
            ];
        } else {
            userContent = cleanPrompt;
        }

        // Assemble the message list: system + rolling channel history + this turn
        const systemPrompt = buildSystemPrompt(message.guildId, message.author.id);
        const history = getChannelHistory(channelId);
        const messages = [
            { role: 'system', content: systemPrompt },
            ...history,
            { role: 'user', content: userContent },
        ];

        const model = hasImages ? VISION_MODEL : TEXT_MODEL;
        const isCode = looksLikeCode(cleanPrompt);

        const response = await mistral.chat.complete({
            model,
            messages,
            maxTokens: isCode ? 1800 : 900,
        });

        const botReply = response.choices[0].message.content || '(không có phản hồi)';

        // Save to short-term rolling history (store plain text, not image blobs, to keep it light)
        pushHistory(channelId, { role: 'user', content: hasImages ? `${cleanPrompt} [đã gửi ${imageUrls.length} ảnh]` : cleanPrompt });
        pushHistory(channelId, { role: 'assistant', content: botReply });
        saveStoreSoon();

        // Periodic "learning" pass — distill durable facts about this user
        user.msgCount += 1;
        if (user.msgCount - user.lastLearn >= LEARN_EVERY) {
            user.lastLearn = user.msgCount;
            learnAboutUser(message.author.id, cleanPrompt); // fire and forget
        }

        // Discord's 2000-char limit — split into chunks instead of truncating
        if (botReply.length <= 2000) {
            await message.reply(botReply);
        } else {
            const chunks = botReply.match(/[\s\S]{1,1990}/g) || [];
            for (const chunk of chunks) {
                await message.channel.send(chunk);
            }
        }
    } catch (error) {
        console.error('Execution Error:', error);
        await message.reply('Đã có lỗi xảy ra khi kết nối tới Mistral AI. Hãy thử lại sau nhé!');
    }
});

// Flush memory to disk on graceful shutdown
process.on('SIGTERM', () => {
    fs.writeFileSync(DATA_FILE, JSON.stringify(store, null, 2));
    process.exit(0);
});

client.login(process.env.DISCORD_TOKEN);
