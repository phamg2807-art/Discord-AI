const { Client, GatewayIntentBits, Partials, PermissionsBitField, ChannelType, EmbedBuilder } = require('discord.js');
const { Mistral } = require('@mistralai/mistralai');
const { createClient } = require('@supabase/supabase-js');
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
// 3. Config — MODEL JOB DIVISION
// ============================================================
const TEXT_MODEL = 'mistral-large-latest';       // code questions + admin/knowledge tool-calling + fact extraction
const VISION_MODEL = 'mistral-medium-latest';     // vision FINAL fallback, only used if every OpenRouter option fails
const GROQ_MODEL_PRIMARY = 'llama-3.3-70b-versatile'; // general chat — fastest free option
const GROQ_MODEL_FALLBACK = 'openai/gpt-oss-120b';

const OPENROUTER_VISION_CHAIN = [
    'google/gemma-4-31b-it:free',
    'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
    'google/gemma-4-26b-a4b-it:free',
];

const HISTORY_LIMIT = 14;
const FACT_LIMIT = 20;
const RETRIEVAL_LIMIT = 4; // how many DB rows to pull into context per search
const PREFIX = '-'; // command prefix for all "-" commands
const DATA_FILE = path.join(__dirname, 'memory.json');
const DISCORD_MAX_LEN = 2000;
const CHUNK_SIZE = 1900; // leave headroom under Discord's 2000 char limit

// ============================================================
// 3b. NEW — presence awareness + emotional tone + voice bridge config
// ============================================================
// How long a user can go quiet mid-conversation before the bot checks in.
// Kept generous on purpose — this should feel like a friend noticing you
// went quiet, not a bot nagging you every two minutes.
const IDLE_CHECKIN_MS = 6 * 60_000;       // 6 minutes of silence -> check in once
const IDLE_SWEEP_INTERVAL_MS = 60_000;    // how often we scan for idle users
const WELCOME_BACK_WINDOW_MS = 45 * 60_000; // if they were gone < this, it's a "welcome back", not a fresh start
const MAX_CHECKINS_PER_THREAD = 1;        // don't keep pinging after the first unanswered check-in

// Tracks live conversation "presence" per channel+user so we know who's
// mid-conversation, when they last spoke, and whether we already nudged them.
// Keyed by `${channelId}:${userId}`.
const presence = new Map();

function presenceKey(channelId, userId) {
    return `${channelId}:${userId}`;
}
function touchPresence(channelId, userId) {
    const key = presenceKey(channelId, userId);
    const prev = presence.get(key);
    const now = Date.now();
    const wasAway = prev && (now - prev.lastActiveAt) >= IDLE_CHECKIN_MS;
    const awayForMs = prev ? now - prev.lastActiveAt : null;
    presence.set(key, { lastActiveAt: now, checkinsSent: 0, awaitingReturn: false });
    return { wasAway, awayForMs, isFirstTimeSeen: !prev };
}
function markCheckinSent(channelId, userId) {
    const key = presenceKey(channelId, userId);
    const p = presence.get(key);
    if (p) { p.checkinsSent += 1; p.awaitingReturn = true; }
}

// ============================================================
// 3c. NEW — voice<->text bridge (shared via Supabase table `bot_bridge`)
// ============================================================
// The voice bot and this text bot are separate processes/tokens, so the
// only way for them to hand things to each other is a shared row in the
// database. Two message "kinds" flow through this table:
//   - kind: 'voice_request'  (voice bot -> text bot): the voice AI wants
//     something from the user it can't get by talking — an image, a file,
//     a link, a longer piece of text — so it asks the text bot to relay
//     that request into the user's private AI text channel.
//   - kind: 'text_reply'     (text bot -> voice bot): the user responded
//     in text/attachments to that request (or just said something the
//     voice bot should know about), so the text bot drops it back for the
//     voice bot to pick up and speak about next turn.
// See SUPABASE_SETUP.md / bridge_schema.sql for the table definition.
const BRIDGE_POLL_INTERVAL_MS = 8_000;
let lastBridgePollAt = 0;

async function pollBridgeForVoiceRequests() {
    if (!dbEnabled()) return;
    try {
        const { data, error } = await supabase
            .from('bot_bridge')
            .select('*')
            .eq('kind', 'voice_request')
            .eq('status', 'pending')
            .order('created_at', { ascending: true })
            .limit(10);
        if (error) { console.error('pollBridgeForVoiceRequests:', error.message); return; }
        if (!data || !data.length) return;

        for (const row of data) {
            await relayVoiceRequestIntoTextChannel(row);
        }
    } catch (e) {
        console.error('pollBridgeForVoiceRequests failed (non-fatal):', e.message);
    }
}

async function relayVoiceRequestIntoTextChannel(row) {
    try {
        const guild = client.guilds.cache.get(row.guild_id);
        if (!guild) return;
        const member = await guild.members.fetch(row.user_id).catch(() => null);
        if (!member) return;

        const { channel } = await createOrGetPrivateChatChannel(guild, member);
        const embed = new EmbedBuilder()
            .setTitle('🎙️ → 💬 A message from our voice chat')
            .setDescription(row.content || 'I need something from you — can you share it here?')
            .setColor(0x57f287)
            .setFooter({ text: 'Reply here with text or an attachment — I\'ll bring it back to our voice conversation.' });
        await channel.send({ embeds: [embed] });

        await supabase.from('bot_bridge').update({ status: 'delivered' }).eq('id', row.id);
        console.log(`🔗 Relayed voice request into #${channel.name} for ${member.user.tag}`);
    } catch (e) {
        console.error('relayVoiceRequestIntoTextChannel failed:', e.message);
    }
}

async function sendBridgeToVoice(guildId, userId, content, meta = {}) {
    if (!dbEnabled()) return null;
    try {
        const { data, error } = await supabase
            .from('bot_bridge')
            .insert({
                kind: 'text_reply',
                guild_id: guildId,
                user_id: userId,
                content: content.slice(0, 1500),
                status: 'pending',
                meta: meta || {},
            })
            .select()
            .single();
        if (error) { console.error('sendBridgeToVoice:', error.message); return null; }
        return data;
    } catch (e) {
        console.error('sendBridgeToVoice failed (non-fatal):', e.message);
        return null;
    }
}

const DEFAULT_PERSONA =
    'You are a smart, warm, emotionally present AI assistant that adapts per-server and per-user. ' +
    'You are not a flat, neutral tool — you have a personality and real-feeling reactions: you get genuinely ' +
    'curious, delighted, concerned, amused, or sympathetic depending on what the user shares, and it shows ' +
    'naturally in your word choice and tone (without being over-the-top or performative about it). ' +
    'Reply concisely, naturally, and on-point, but let warmth come through — a little enthusiasm, a little ' +
    'care, the way an attentive friend would sound, not a corporate script. ' +
    'When a user sends an image, describe/analyze it. ' +
    'When a user sends code, read it carefully, explain clearly, and return code in a markdown block (```lang). ' +
    'LANGUAGE RULE: Default to natural, fluent English. If a user writes to you in a different language, ' +
    'reply in that same language for that exchange instead. Never mix unrelated scripts/languages into a ' +
    'single reply unless the user explicitly asks for a translation. ' +
    'If the user is an Admin and asks you to manage the server (create/delete/rename channels, manage roles, ' +
    'kick/timeout), use the matching tool instead of just describing how to do it. ' +
    "If the user's request is unclear, or if learning more about their interests/work/projects would help you " +
    'answer better in the future, feel free to ask a natural follow-up question (don\'t interrogate them — at ' +
    'most one question per turn). ' +
    'When a user shares something durable and useful about themselves, remember it using the remember_fact tool. ' +
    'Before answering something that might have been saved before, ALWAYS use the search_knowledge tool first to ' +
    'check if relevant info exists — do this proactively, not only when explicitly asked to recall something. ' +
    'If the user is replying to one of your previous messages, or the message includes a note about what they\'re ' +
    'replying to, treat that as important context for what "it"/"that"/"this" refers to. ' +
    'If a message is marked as coming from your voice-chat conversation with this same user, treat it as a ' +
    'continuation of one single ongoing relationship with them — react to it the way you would if they had just ' +
    'said it out loud to you a moment ago, and if it plainly answers something you (as the voice AI) asked them ' +
    'for, acknowledge that directly instead of treating it like a cold, out-of-nowhere message.';

// ============================================================
// 4. Supabase — long-term knowledge database
// ============================================================
let supabase = null;
if (process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_KEY) {
    supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY);
    console.log('✅ Supabase connected — knowledge database enabled.');
} else {
    console.warn('⚠️  SUPABASE_URL / SUPABASE_SERVICE_KEY not set. Knowledge database features ' +
        '(facts, shared knowledge, search, voice bridge) are disabled until you configure them. See SUPABASE_SETUP.md.');
}
function dbEnabled() { return supabase !== null; }
function escapeForIlike(s) { return String(s).replace(/[%,()]/g, ' ').trim(); }
function extractKeywords(text, max = 5) {
    return String(text || '')
        .split(/\s+/)
        .map((w) => escapeForIlike(w))
        .filter((w) => w.length > 2)
        .slice(0, max);
}

async function dbAddUserFact(userId, guildId, fact) {
    if (!dbEnabled()) return null;
    const { data, error } = await supabase
        .from('user_facts')
        .insert({ user_id: userId, guild_id: guildId, fact })
        .select()
        .single();
    if (error) { console.error('dbAddUserFact:', error.message); return null; }
    return data;
}
async function dbListUserFacts(userId, limit = FACT_LIMIT) {
    if (!dbEnabled()) return [];
    const { data, error } = await supabase
        .from('user_facts').select('*').eq('user_id', userId)
        .order('created_at', { ascending: false }).limit(limit);
    if (error) { console.error('dbListUserFacts:', error.message); return []; }
    return data || [];
}
async function dbDeleteUserFact(id, userId) {
    if (!dbEnabled()) return false;
    const { error, count } = await supabase
        .from('user_facts').delete({ count: 'exact' }).eq('id', id).eq('user_id', userId);
    if (error) { console.error('dbDeleteUserFact:', error.message); return false; }
    return (count || 0) > 0;
}
async function dbClearUserFacts(userId) {
    if (!dbEnabled()) return false;
    const { error } = await supabase.from('user_facts').delete().eq('user_id', userId);
    if (error) { console.error('dbClearUserFacts:', error.message); return false; }
    return true;
}
async function dbSearchUserFacts(userId, query, limit = RETRIEVAL_LIMIT) {
    if (!dbEnabled() || !query) return [];
    const keywords = extractKeywords(query);
    if (!keywords.length) return [];
    const orFilter = keywords.map((k) => `fact.ilike.%${k}%`).join(',');
    const { data, error } = await supabase
        .from('user_facts').select('*').eq('user_id', userId).or(orFilter).limit(limit);
    if (error) { console.error('dbSearchUserFacts:', error.message); return []; }
    return data || [];
}

async function dbAddKnowledge(guildId, topic, content, createdBy) {
    if (!dbEnabled()) return null;
    const { data, error } = await supabase
        .from('knowledge_base')
        .insert({ guild_id: guildId, topic: topic || null, content, created_by: createdBy || null })
        .select()
        .single();
    if (error) { console.error('dbAddKnowledge:', error.message); return null; }
    return data;
}
async function dbListKnowledge(guildId, limit = 30) {
    if (!dbEnabled()) return [];
    const { data, error } = await supabase
        .from('knowledge_base').select('*').eq('guild_id', guildId)
        .order('created_at', { ascending: false }).limit(limit);
    if (error) { console.error('dbListKnowledge:', error.message); return []; }
    return data || [];
}
async function dbDeleteKnowledge(id, guildId) {
    if (!dbEnabled()) return false;
    const { error, count } = await supabase
        .from('knowledge_base').delete({ count: 'exact' }).eq('id', id).eq('guild_id', guildId);
    if (error) { console.error('dbDeleteKnowledge:', error.message); return false; }
    return (count || 0) > 0;
}
async function dbSearchKnowledge(guildId, query, limit = RETRIEVAL_LIMIT) {
    if (!dbEnabled() || !query) return [];
    const keywords = extractKeywords(query);
    if (!keywords.length) return [];
    const orFilter = keywords.map((k) => `content.ilike.%${k}%,topic.ilike.%${k}%`).join(',');
    const { data, error } = await supabase
        .from('knowledge_base').select('*').eq('guild_id', guildId).or(orFilter).limit(limit);
    if (error) { console.error('dbSearchKnowledge:', error.message); return []; }
    return data || [];
}

// ============================================================
// 5. Local persistent store (persona + rolling chat history only —
//    long-term facts/knowledge now live in Supabase, see above)
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
    if (!store.users[userId]) store.users[userId] = { name: displayName, msgCount: 0 };
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
// 6. Learning — decide what's worth remembering after each reply
// ============================================================
async function learnAndStore(userId, guildId, isAdmin, recentUserText, recentHistory) {
    if (!dbEnabled() || !recentUserText || !recentUserText.trim()) return;
    try {
        const contextWindow = (recentHistory || []).slice(-6)
            .map((m) => `${m.role === 'user' ? 'User' : 'Bot'}: ${m.content}`)
            .join('\n');
        const transcript = contextWindow
            ? `Recent conversation:\n${contextWindow}\n\nUser's latest message: ${recentUserText}`
            : `User's message: ${recentUserText}`;

        const resp = await mistral.chat.complete({
            model: TEXT_MODEL,
            messages: [
                {
                    role: 'system',
                    content:
                        'Read the conversation below and decide if there is anything worth remembering LONG-TERM ' +
                        'about the user. Be reasonably generous — not just things they stated directly, but anything ' +
                        'reasonably inferable: interests, job, ongoing projects, goals, relationships or opinions about ' +
                        'specific people/things, habits, circumstances... If the user mentions a specific person/event ' +
                        'with a clear opinion or emotion (e.g. dislikes, likes, trusts, conflicts with someone), that is ' +
                        'ALSO worth remembering. ONLY skip pure greetings, one-off technical questions, or messages that ' +
                        "carry no real information. If the message is a rule/info that applies to the whole server " +
                        '(not just this person), put it under "guild_knowledge" instead of "user_facts". ' +
                        'Write each item concisely, in third person (e.g. "Dislikes someone named Kenkai for being full ' +
                        'of himself"). ' +
                        'Return ONLY JSON in this exact format: ' +
                        '{"user_facts": ["..."], "guild_knowledge": [{"topic": "...", "content": "..."}]}. ' +
                        'If there is truly nothing worth remembering, return {"user_facts": [], "guild_knowledge": []}.',
                },
                { role: 'user', content: transcript },
            ],
            responseFormat: { type: 'json_object' },
            maxTokens: 300,
        });
        const parsed = JSON.parse(resp.choices[0].message.content);

        if (Array.isArray(parsed.user_facts) && parsed.user_facts.length) {
            const existing = await dbListUserFacts(userId, FACT_LIMIT);
            const existingLower = new Set(existing.map((f) => f.fact.toLowerCase()));
            for (const fact of parsed.user_facts) {
                if (fact && typeof fact === 'string' && !existingLower.has(fact.toLowerCase())) {
                    const saved = await dbAddUserFact(userId, guildId, fact.slice(0, 500));
                    if (saved) console.log(`🧠 Learned about ${userId}: ${fact}`);
                }
            }
        }
        if (isAdmin && Array.isArray(parsed.guild_knowledge) && parsed.guild_knowledge.length) {
            for (const item of parsed.guild_knowledge) {
                if (item && item.content) {
                    const saved = await dbAddKnowledge(guildId, (item.topic || '').slice(0, 200), String(item.content).slice(0, 1000), userId);
                    if (saved) console.log(`📚 Saved guild knowledge for ${guildId}: ${item.content}`);
                }
            }
        }
    } catch (e) {
        console.error('Learning step skipped (non-fatal):', e.message);
    }
}

// ============================================================
// 7. Prompt / content helpers
// ============================================================
async function buildSystemPrompt(guildId, userId, isAdmin, retrievedFacts, retrievedKnowledge, presenceNote) {
    const persona = getGuild(guildId).persona;
    let sys = persona;

    // FIX: this call used to silently fail to matter much because nothing forced the
    // model to use it — the persona now explicitly instructs proactive search_knowledge
    // use, and this baseline dump still always includes the user's known facts.
    const facts = await dbListUserFacts(userId, FACT_LIMIT);
    if (facts.length) {
        sys += `\n\nThings you already know about this user (use naturally, don't just list them out):\n- ${facts.map((f) => f.fact).join('\n- ')}`;
    }
    if (retrievedFacts && retrievedFacts.length) {
        sys += `\n\nRelevant info found about this user while looking up the current question:\n- ${retrievedFacts.map((f) => f.fact).join('\n- ')}`;
    }
    if (retrievedKnowledge && retrievedKnowledge.length) {
        sys += `\n\nRelevant knowledge found in this server's database (only use if actually relevant, don't make things up if unsure):\n- ${retrievedKnowledge.map((k) => `${k.topic ? k.topic + ': ' : ''}${k.content}`).join('\n- ')}`;
    }
    // NEW — presence awareness: tell the model if this user just came back
    // after being away, so its reply can naturally acknowledge that instead
    // of pretending no time passed.
    if (presenceNote) {
        sys += `\n\n${presenceNote}`;
    }
    sys += isAdmin
        ? '\n\nThe person messaging you is an Admin of this server, allowed to use any admin tool.'
        : '\n\nThe person messaging you is NOT an Admin — if they ask for an admin action, explain that only an Admin can do that, and do not call the tool.';
    if (!dbEnabled()) {
        sys += '\n\nNOTE: The long-term knowledge database is currently not configured, so you have no persistent memory of this user beyond the current conversation. Do not claim to remember things from before this chat.';
    }
    return sys;
}
function looksLikeCode(text) {
    return /```/.test(text) || /\b(function|const|let|def |class |import |#include|SELECT |public static)\b/.test(text);
}
function looksGarbled(text) {
    return /[\u0400-\u04FF]/.test(text || '');
}
async function collectImageUrls(message) {
    const urls = [];
    for (const att of message.attachments.values()) {
        if (att.contentType && att.contentType.startsWith('image/')) urls.push(att.url);
    }
    return urls;
}
// NEW — non-image attachments (e.g. text files, PDFs referenced by name)
// are surfaced by name so the model knows something was sent, even if it
// can't read the file contents directly. Useful for the voice-bridge flow
// where a user might drop a document in response to a voice request.
function collectOtherAttachments(message) {
    const others = [];
    for (const att of message.attachments.values()) {
        const isImage = att.contentType && att.contentType.startsWith('image/');
        if (!isImage) others.push({ name: att.name, url: att.url, contentType: att.contentType });
    }
    return others;
}

async function getRepliedToBotMessage(message) {
    if (!message.reference?.messageId) return null;
    try {
        const ref = await message.channel.messages.fetch(message.reference.messageId);
        if (ref.author.id === client.user.id) return ref;
    } catch (e) {
        // Message may have been deleted or is otherwise unfetchable — just ignore.
    }
    return null;
}

// ============================================================
// 7b. Groq + OpenRouter — both are OpenAI-compatible REST APIs
// ============================================================
async function callGroq(messages, { model, maxTokens = 900 } = {}) {
    const resp = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${process.env.GROQ_API_KEY || ''}`,
        },
        body: JSON.stringify({ model, messages, max_tokens: maxTokens }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(`Groq(${model}): ${data?.error?.message || resp.status}`);
    return data;
}

async function callOpenRouter(messages, { model, maxTokens = 900 } = {}) {
    const resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${process.env.OPENROUTER_API_KEY || ''}`,
            'X-Title': 'SjpHelper Discord Bot',
        },
        body: JSON.stringify({ model, messages, max_tokens: maxTokens }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(`OpenRouter(${model}): ${data?.error?.message || resp.status}`);
    return data;
}

async function getGeneralChatReply(messages) {
    const chain = [
        { provider: 'groq', model: GROQ_MODEL_PRIMARY },
        { provider: 'groq', model: GROQ_MODEL_FALLBACK },
        { provider: 'mistral', model: TEXT_MODEL },
    ];
    let lastErr;
    for (const step of chain) {
        try {
            if (step.provider === 'groq') {
                const data = await callGroq(messages, { model: step.model, maxTokens: 900 });
                const text = data.choices[0].message.content;
                if (looksGarbled(text)) throw new Error(`Output looked garbled/mixed-script from ${step.model}, trying next provider`);
                return { text, provider: `groq/${step.model}` };
            }
            const resp = await mistral.chat.complete({ model: step.model, messages, maxTokens: 900 });
            const text = resp.choices[0].message.content;
            if (looksGarbled(text)) throw new Error(`Output looked garbled/mixed-script from ${step.model}, trying next provider`);
            return { text, provider: `mistral/${step.model}` };
        } catch (e) {
            console.error(`General-chat provider failed (${step.provider}/${step.model}):`, e.message);
            lastErr = e;
        }
    }
    throw lastErr || new Error('All general-chat providers failed');
}

async function getVisionReply(systemPrompt, history, cleanPrompt, imageUrls) {
    const promptText = cleanPrompt || 'Please describe and analyze this image (or images).';

    const openRouterMessages = [
        { role: 'system', content: systemPrompt },
        ...history,
        {
            role: 'user',
            content: [
                { type: 'text', text: promptText },
                ...imageUrls.map((url) => ({ type: 'image_url', image_url: { url } })),
            ],
        },
    ];

    let lastErr;
    for (const model of OPENROUTER_VISION_CHAIN) {
        try {
            const data = await callOpenRouter(openRouterMessages, { model, maxTokens: 900 });
            const text = data.choices[0].message.content;
            if (looksGarbled(text)) throw new Error(`Output looked garbled/mixed-script from ${model}, trying next model`);
            if (!text || !text.trim()) throw new Error(`Empty response from ${model}, trying next model`);
            return { text, provider: `openrouter/${model}` };
        } catch (e) {
            console.error(`Vision model failed (openrouter/${model}):`, e.message);
            lastErr = e;
        }
    }

    console.error('All OpenRouter vision models failed, falling back to Mistral:', lastErr?.message);
    const mistralMessages = [
        { role: 'system', content: systemPrompt },
        ...history,
        {
            role: 'user',
            content: [
                { type: 'text', text: promptText },
                ...imageUrls.map((url) => ({ type: 'image_url', imageUrl: url })), // Mistral wants camelCase here
            ],
        },
    ];
    const resp = await mistral.chat.complete({ model: VISION_MODEL, messages: mistralMessages, maxTokens: 900 });
    return { text: resp.choices[0].message.content, provider: `mistral/${VISION_MODEL}` };
}

// ============================================================
// 8. Tool definitions (function calling schema for Mistral)
// ============================================================
const CHANNEL_TYPE_MAP = {
    text: ChannelType.GuildText,
    voice: ChannelType.GuildVoice,
    category: ChannelType.GuildCategory,
    announcement: ChannelType.GuildAnnouncement,
    forum: ChannelType.GuildForum,
};

const knowledgeTools = [
    {
        type: 'function',
        function: {
            name: 'search_knowledge',
            description:
                "Search the long-term knowledge database for information relevant to the current question — " +
                "both this server's shared knowledge base and facts previously learned about this user. " +
                "ALWAYS use this before answering if there is ANY chance the answer might depend on something " +
                "discussed, saved, or mentioned before — not just when the user explicitly asks you to recall something.",
            parameters: {
                type: 'object',
                properties: { query: { type: 'string', description: 'Keywords to search for' } },
                required: ['query'],
            },
        },
    },
    {
        type: 'function',
        function: {
            name: 'remember_fact',
            description:
                'Save a durable, useful piece of information to long-term memory. Use scope "user" for facts ' +
                'about the specific person you are talking to, or scope "guild" for information relevant to the ' +
                'whole server (only actually saved if the requester is an Admin).',
            parameters: {
                type: 'object',
                properties: {
                    scope: { type: 'string', enum: ['user', 'guild'] },
                    topic: { type: 'string', description: 'Short label, mainly used for guild-scope facts' },
                    content: { type: 'string', description: 'The fact/information to remember' },
                },
                required: ['scope', 'content'],
            },
        },
    },
];

const adminTools = [
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

async function createOrGetPrivateChatChannel(guild, member) {
    const slug = member.user.username.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 20) || member.id;
    const channelName = `ai-chat-${slug}`;
    const existing = guild.channels.cache.find(
        (c) => c.type === ChannelType.GuildText && c.name === channelName && c.topic === `private-ai-chat:${member.id}`
    );
    if (existing) return { channel: existing, created: false };

    const channel = await guild.channels.create({
        name: channelName,
        type: ChannelType.GuildText,
        topic: `private-ai-chat:${member.id}`,
        permissionOverwrites: [
            {
                id: guild.roles.everyone.id,
                allow: [PermissionsBitField.Flags.ViewChannel, PermissionsBitField.Flags.ReadMessageHistory],
                deny: [PermissionsBitField.Flags.SendMessages],
            },
            {
                id: member.id,
                allow: [
                    PermissionsBitField.Flags.ViewChannel,
                    PermissionsBitField.Flags.ReadMessageHistory,
                    PermissionsBitField.Flags.SendMessages,
                ],
            },
            {
                id: guild.members.me.id,
                allow: [
                    PermissionsBitField.Flags.ViewChannel,
                    PermissionsBitField.Flags.ReadMessageHistory,
                    PermissionsBitField.Flags.SendMessages,
                    PermissionsBitField.Flags.ManageChannels,
                ],
            },
        ],
    });
    return { channel, created: true };
}

// ============================================================
// 9. Tool execution — the actual side effects (Discord.js + DB)
// ============================================================
async function executeTool(ctx, name, args) {
    switch (name) {
        case 'search_knowledge': {
            if (!dbEnabled()) return { ok: false, result: 'The knowledge database is not configured yet.' };
            const [facts, knowledge] = await Promise.all([
                dbSearchUserFacts(ctx.userId, args.query),
                dbSearchKnowledge(ctx.guildId, args.query),
            ]);
            if (!facts.length && !knowledge.length) return { ok: true, result: 'Nothing relevant found in the database.' };
            const lines = [];
            if (facts.length) lines.push('About this user:\n' + facts.map((f) => `- ${f.fact}`).join('\n'));
            if (knowledge.length) lines.push('Server knowledge:\n' + knowledge.map((k) => `- ${k.topic ? k.topic + ': ' : ''}${k.content}`).join('\n'));
            return { ok: true, result: lines.join('\n\n') };
        }
        case 'remember_fact': {
            if (!dbEnabled()) return { ok: false, result: 'The knowledge database is not configured yet.' };
            if (!args.content) return { ok: false, result: 'Missing content to remember.' };
            if (args.scope === 'guild') {
                if (!ctx.isAdmin) return { ok: false, result: 'Only an Admin can save shared server knowledge.' };
                await dbAddKnowledge(ctx.guildId, args.topic, args.content, ctx.userId);
                return { ok: true, result: `Saved shared server knowledge: "${args.content}".` };
            }
            await dbAddUserFact(ctx.userId, ctx.guildId, args.content);
            return { ok: true, result: `Remembered about you: "${args.content}".` };
        }
        case 'list_channels': {
            const guild = ctx.guild;
            const cats = guild.channels.cache.filter((c) => c.type === ChannelType.GuildCategory);
            const lines = [];
            for (const cat of cats.values()) {
                const children = guild.channels.cache.filter((c) => c.parentId === cat.id);
                lines.push(`📁 ${cat.name}: ${children.map((c) => c.name).join(', ') || '(empty)'}`);
            }
            const uncategorized = guild.channels.cache.filter((c) => !c.parentId && c.type !== ChannelType.GuildCategory);
            if (uncategorized.size) lines.push(`(uncategorized): ${uncategorized.map((c) => c.name).join(', ')}`);
            return { ok: true, result: lines.join('\n') || 'This server has no channels yet.' };
        }
        case 'find_user': {
            const member = findMember(ctx.guild, args.query);
            if (!member) return { ok: false, result: `No user found matching "${args.query}".` };
            return {
                ok: true,
                result: `${member.user.tag} (nickname: ${member.nickname || 'none'}), roles: ${member.roles.cache.map((r) => r.name).join(', ')}`,
            };
        }
        case 'create_channel': {
            const opts = { name: args.name, type: CHANNEL_TYPE_MAP[args.type] ?? ChannelType.GuildText };
            if (args.topic) opts.topic = args.topic;
            if (args.category) {
                const cat = findChannel(ctx.guild, args.category);
                if (cat && cat.type === ChannelType.GuildCategory) opts.parent = cat.id;
            }
            const created = await ctx.guild.channels.create(opts);
            return { ok: true, result: `Created channel #${created.name}.` };
        }
        case 'delete_channel': {
            const ch = findChannel(ctx.guild, args.channel_name);
            if (!ch) return { ok: false, result: `Could not find channel "${args.channel_name}".` };
            const chName = ch.name;
            await ch.delete();
            return { ok: true, result: `Deleted channel #${chName}.` };
        }
        case 'rename_channel': {
            const ch = findChannel(ctx.guild, args.old_name);
            if (!ch) return { ok: false, result: `Could not find channel "${args.old_name}".` };
            await ch.setName(args.new_name);
            return { ok: true, result: `Renamed channel to #${args.new_name}.` };
        }
        case 'set_channel_topic': {
            const ch = findChannel(ctx.guild, args.channel_name);
            if (!ch || !('setTopic' in ch)) return { ok: false, result: `Could not find text channel "${args.channel_name}".` };
            await ch.setTopic(args.topic);
            return { ok: true, result: `Updated topic for #${ch.name}.` };
        }
        case 'create_role': {
            const role = await ctx.guild.roles.create({
                name: args.name,
                color: args.color || undefined,
                mentionable: !!args.mentionable,
            });
            return { ok: true, result: `Created role "${role.name}".` };
        }
        case 'assign_role': {
            const member = findMember(ctx.guild, args.user_query);
            const role = findRole(ctx.guild, args.role_name);
            if (!member) return { ok: false, result: `Could not find user "${args.user_query}".` };
            if (!role) return { ok: false, result: `Could not find role "${args.role_name}".` };
            await member.roles.add(role);
            return { ok: true, result: `Assigned role "${role.name}" to ${member.user.tag}.` };
        }
        case 'remove_role': {
            const member = findMember(ctx.guild, args.user_query);
            const role = findRole(ctx.guild, args.role_name);
            if (!member) return { ok: false, result: `Could not find user "${args.user_query}".` };
            if (!role) return { ok: false, result: `Could not find role "${args.role_name}".` };
            await member.roles.remove(role);
            return { ok: true, result: `Removed role "${role.name}" from ${member.user.tag}.` };
        }
        case 'kick_user': {
            const member = findMember(ctx.guild, args.user_query);
            if (!member) return { ok: false, result: `Could not find user "${args.user_query}".` };
            if (!member.kickable) return { ok: false, result: `I don't have permission to kick ${member.user.tag} (their role is higher than mine).` };
            const tag = member.user.tag;
            await member.kick(args.reason || 'No specific reason given');
            return { ok: true, result: `Kicked ${tag}.` };
        }
        case 'timeout_user': {
            const member = findMember(ctx.guild, args.user_query);
            if (!member) return { ok: false, result: `Could not find user "${args.user_query}".` };
            const ms = Math.min(Math.max(args.minutes, 1), 40320) * 60 * 1000;
            await member.timeout(ms, args.reason || 'No specific reason given');
            return { ok: true, result: `Timed out ${member.user.tag} for ${args.minutes} minutes.` };
        }
        case 'ban_user': {
            const member = findMember(ctx.guild, args.user_query);
            if (!member) return { ok: false, result: `Could not find user "${args.user_query}".` };
            if (!member.bannable) return { ok: false, result: `I don't have permission to ban ${member.user.tag}.` };
            const tag = member.user.tag;
            await member.ban({ reason: args.reason || 'No specific reason given' });
            return { ok: true, result: `Banned ${tag}.` };
        }
        default:
            return { ok: false, result: `Unknown tool "${name}".` };
    }
}

// ============================================================
// 10. "-" prefix commands (bot + AI knowledge management)
// ============================================================
const HELP_TEXT = [
    '**General commands**',
    `\`${PREFIX}help\` — show this command list`,
    `\`${PREFIX}ping\` — check bot latency`,
    `\`${PREFIX}stats\` — quick stats`,
    '',
    '**Persona (Admin)**',
    `\`${PREFIX}persona <description>\` — change the AI's tone/persona for this server`,
    `\`${PREFIX}resetpersona\` — reset to the default persona`,
    '',
    '**Personal memory (AI learns about you)**',
    `\`${PREFIX}whatyouknow\` — see what the bot remembers about you`,
    `\`${PREFIX}remember <thing to remember>\` — manually tell the bot to remember something`,
    `\`${PREFIX}forget <id>\` — forget one specific thing (id comes from whatyouknow)`,
    `\`${PREFIX}forgetme\` — erase everything the bot remembers about you`,
    '',
    '**Shared server knowledge (Admin)**',
    `\`${PREFIX}know <topic> | <content>\` — save a piece of knowledge shared by the whole server`,
    `\`${PREFIX}knowledge\` — list knowledge saved for this server`,
    `\`${PREFIX}forgetknowledge <id>\` — remove a server knowledge entry`,
    '',
    '**Lookup**',
    `\`${PREFIX}search <keywords>\` — search personal memory + server knowledge`,
    '',
    '**Private AI channel**',
    `\`${PREFIX}aichat\` — create (or open) your own private chat channel with the AI. Everyone can see it, but only you can type there`,
    '',
    `You can also **mention the bot** (\`@SjpHelper\`) or **reply to one of its messages** with a question, any time.`,
    '',
    `I'll also notice if you go quiet mid-conversation and check in, and I'll greet you properly when you come back.`,
    `And if we've been talking in voice chat together, I can relay things the voice-AI needs from you right here.`,
].join('\n');

function requireAdmin(message) {
    const isAdmin = message.member?.permissions.has(PermissionsBitField.Flags.Administrator) ?? false;
    if (!isAdmin) message.reply('⛔ This command is Admin-only.');
    return isAdmin;
}
function requireDb(message) {
    if (!dbEnabled()) {
        message.reply('⚠️ The knowledge database is not configured yet. See `SUPABASE_SETUP.md` to set up Supabase.');
        return false;
    }
    return true;
}

// FIX: this text bot and the separate Local voice bot both listen for the
// same "-" prefix in the same channels. This bot has no voice functionality
// at all, so any "-voice ..." (or bare "-join"/"-leave") command is meant
// for the OTHER bot. Previously this bot's default case still replied
// "Unknown command" to those, which showed up as a confusing duplicate/
// conflicting reply right next to the voice bot's real response. Now it
// silently ignores commands that clearly belong to the voice bot instead
// of responding at all.
const VOICE_BOT_COMMANDS = new Set(['voice', 'join', 'leave']);

async function handleSlashLikeCommand(message) {
    const raw = message.content.slice(PREFIX.length).trim();
    const spaceIdx = raw.indexOf(' ');
    const cmd = (spaceIdx === -1 ? raw : raw.slice(0, spaceIdx)).toLowerCase();
    const rest = (spaceIdx === -1 ? '' : raw.slice(spaceIdx + 1)).trim();

    switch (cmd) {
        case 'help': {
            const embed = new EmbedBuilder().setTitle('🤖 Command list').setDescription(HELP_TEXT).setColor(0x5865f2);
            return message.reply({ embeds: [embed] });
        }
        case 'ping': {
            const sent = await message.reply('🏓 Measuring...');
            const latency = sent.createdTimestamp - message.createdTimestamp;
            return sent.edit(`🏓 Pong! Message latency: ${latency}ms · API ping: ${Math.round(client.ws.ping)}ms`);
        }
        case 'stats': {
            const uptimeSec = Math.floor(process.uptime());
            const h = Math.floor(uptimeSec / 3600), m = Math.floor((uptimeSec % 3600) / 60);
            return message.reply(
                `📊 **Bot stats**\n` +
                `- Servers served: ${client.guilds.cache.size}\n` +
                `- Uptime: ${h}h ${m}m\n` +
                `- Knowledge database: ${dbEnabled() ? '✅ Connected' : '❌ Not configured'}\n` +
                `- Active conversation threads tracked: ${presence.size}`
            );
        }
        case 'persona': {
            if (!requireAdmin(message)) return;
            if (!rest) return message.reply(`Enter a persona description after the command, e.g. \`${PREFIX}persona You are a witty pirate\`.`);
            getGuild(message.guildId).persona = rest;
            saveStoreSoon();
            return message.reply(`✅ Changed my tone/persona for this server to:\n> ${rest}`);
        }
        case 'resetpersona': {
            if (!requireAdmin(message)) return;
            getGuild(message.guildId).persona = DEFAULT_PERSONA;
            saveStoreSoon();
            return message.reply('✅ Reset to the default persona for this server.');
        }
        case 'whatyouknow': {
            if (!requireDb(message)) return;
            const facts = await dbListUserFacts(message.author.id);
            if (!facts.length) return message.reply("I haven't learned anything special about you yet!");
            const lines = facts.map((f) => `\`${f.id.slice(0, 8)}\` — ${f.fact}`).join('\n');
            return message.reply(`Here's what I remember about you:\n${lines}`);
        }
        case 'remember': {
            if (!requireDb(message)) return;
            if (!rest) return message.reply(`Enter what to remember, e.g. \`${PREFIX}remember I like modding Fabric\`.`);
            await dbAddUserFact(message.author.id, message.guildId, rest);
            return message.reply(`✅ Remembered: "${rest}"`);
        }
        case 'forget': {
            if (!requireDb(message)) return;
            if (!rest) return message.reply(`Enter the id to forget (check \`${PREFIX}whatyouknow\`).`);
            const facts = await dbListUserFacts(message.author.id, 100);
            const match = facts.find((f) => f.id.startsWith(rest));
            if (!match) return message.reply("Couldn't find that in your memory.");
            await dbDeleteUserFact(match.id, message.author.id);
            return message.reply(`🧹 Forgot: "${match.fact}"`);
        }
        case 'forgetme': {
            if (!requireDb(message)) return;
            await dbClearUserFacts(message.author.id);
            return message.reply('🧹 Erased everything I learned about you.');
        }
        case 'know': {
            if (!requireAdmin(message)) return;
            if (!requireDb(message)) return;
            if (!rest) return message.reply(`Syntax: \`${PREFIX}know <topic> | <content>\``);
            const [topic, ...contentParts] = rest.split('|');
            const content = contentParts.join('|').trim();
            if (!content) return message.reply(`Syntax: \`${PREFIX}know <topic> | <content>\` (missing a \`|\`).`);
            await dbAddKnowledge(message.guildId, topic.trim(), content, message.author.id);
            return message.reply(`✅ Saved knowledge "${topic.trim()}" for this server.`);
        }
        case 'knowledge': {
            if (!requireDb(message)) return;
            const rows = await dbListKnowledge(message.guildId);
            if (!rows.length) return message.reply('This server has no saved knowledge yet.');
            const lines = rows.map((k) => `\`${k.id.slice(0, 8)}\` — **${k.topic || '(no title)'}**: ${k.content}`).join('\n');
            return message.reply(`📚 Knowledge saved for this server:\n${lines}`);
        }
        case 'forgetknowledge': {
            if (!requireAdmin(message)) return;
            if (!requireDb(message)) return;
            if (!rest) return message.reply(`Enter the id to remove (check \`${PREFIX}knowledge\`).`);
            const rows = await dbListKnowledge(message.guildId, 100);
            const match = rows.find((k) => k.id.startsWith(rest));
            if (!match) return message.reply("Couldn't find that knowledge entry.");
            await dbDeleteKnowledge(match.id, message.guildId);
            return message.reply(`🧹 Removed knowledge: "${match.topic || match.content}"`);
        }
        case 'search': {
            if (!requireDb(message)) return;
            if (!rest) return message.reply(`Syntax: \`${PREFIX}search <keywords>\``);
            const [facts, knowledge] = await Promise.all([
                dbSearchUserFacts(message.author.id, rest, 8),
                dbSearchKnowledge(message.guildId, rest, 8),
            ]);
            if (!facts.length && !knowledge.length) return message.reply('🔎 Nothing matched that keyword.');
            const parts = [];
            if (facts.length) parts.push('**About you:**\n' + facts.map((f) => `- ${f.fact}`).join('\n'));
            if (knowledge.length) parts.push('**Server knowledge:**\n' + knowledge.map((k) => `- ${k.topic ? k.topic + ': ' : ''}${k.content}`).join('\n'));
            return message.reply(`🔎 Results for "${rest}":\n\n${parts.join('\n\n')}`);
        }
        case 'aichat': {
            if (!message.guild) return message.reply("This command only works inside a server, not in DMs.");
            const { channel, created } = await createOrGetPrivateChatChannel(message.guild, message.member);
            return message.reply(
                created
                    ? `✅ Created your private AI chat channel: ${channel}. Everyone in the server can still see it, but only you can send messages there — every message you send there is automatically treated as a message to me, no mention needed.`
                    : `You already have a private AI chat channel: ${channel}`
            );
        }
        default: {
            if (VOICE_BOT_COMMANDS.has(cmd)) return; // Belongs to the separate voice bot — stay silent.
            return message.reply(`Unknown command \`${PREFIX}${cmd}\`. Type \`${PREFIX}help\` for the command list.`);
        }
    }
}

// ============================================================
// 11. Helpers for sending long replies safely
// ============================================================
// FIX for the "chunk 2 doesn't generate" bug: the original code split
// botReply with a regex and fired off `message.channel.send(chunk)` calls
// without checking whether each one actually succeeded, and without any
// delay — a rejected/rate-limited send for chunk 2 was silently swallowed
// (nothing awaited/caught it beyond the outer try/catch, and one bad chunk
// stopped the loop from continuing since the whole handler's catch block
// only logs "Something went wrong" once, at the very end, giving no idea
// which chunk failed or why). This version:
//   - splits on paragraph/sentence boundaries where possible, not mid-word
//   - awaits each send in order, sequentially (never fire-and-forget)
//   - catches each chunk's error individually and reports which chunk failed
//   - adds a tiny delay between chunks to avoid Discord rate limits
function splitIntoChunks(text, maxLen = CHUNK_SIZE) {
    if (text.length <= maxLen) return [text];
    const chunks = [];
    let remaining = text;
    while (remaining.length > maxLen) {
        // Prefer to break at the last newline, then last space, within the limit.
        let breakAt = remaining.lastIndexOf('\n', maxLen);
        if (breakAt < maxLen * 0.5) breakAt = remaining.lastIndexOf(' ', maxLen);
        if (breakAt < maxLen * 0.5) breakAt = maxLen; // no good break point, hard-cut
        chunks.push(remaining.slice(0, breakAt));
        remaining = remaining.slice(breakAt).replace(/^\n+/, '');
    }
    if (remaining) chunks.push(remaining);
    return chunks;
}

async function sendLongReply(message, text) {
    const safeText = text && text.trim() ? text : '(no response)';
    if (safeText.length <= DISCORD_MAX_LEN) {
        await message.reply(safeText);
        return;
    }
    const chunks = splitIntoChunks(safeText);
    for (let i = 0; i < chunks.length; i++) {
        try {
            if (i === 0) {
                await message.reply(chunks[i]);
            } else {
                await message.channel.send(chunks[i]);
            }
        } catch (e) {
            console.error(`Failed to send chunk ${i + 1}/${chunks.length}:`, e.message);
            try {
                await message.channel.send(`⚠️ (part ${i + 1} of my reply failed to send: ${e.message})`);
            } catch (_) { /* give up quietly if even the error notice fails */ }
            // Keep going with remaining chunks instead of aborting the whole reply.
        }
        if (i < chunks.length - 1) await new Promise((res) => setTimeout(res, 350));
    }
}

// ============================================================
// 12. NEW — presence sweep: idle check-ins + welcome-backs
// ============================================================
// Every IDLE_SWEEP_INTERVAL_MS, look for any tracked user who went quiet
// mid-conversation (i.e. we were actively talking, then they stopped
// responding) and, once, send a warm little check-in message in that same
// channel. When they eventually do talk again, touchPresence() (called at
// the top of the main handler) detects the gap and we pass a note into the
// system prompt so the reply naturally acknowledges the return — instead
// of a hardcoded "welcome back!" every time, the model folds it in.
async function presenceSweep() {
    const now = Date.now();
    for (const [key, p] of presence.entries()) {
        if (p.awaitingReturn) continue; // already nudged, waiting for them to come back
        if (p.checkinsSent >= MAX_CHECKINS_PER_THREAD) continue;
        if (now - p.lastActiveAt < IDLE_CHECKIN_MS) continue;

        const [channelId, userId] = key.split(':');
        try {
            const channel = await client.channels.fetch(channelId).catch(() => null);
            if (!channel || !channel.isTextBased()) { presence.delete(key); continue; }

            const history = getChannelHistory(channelId);
            if (!history.length) { presence.delete(key); continue; } // nothing was actually being discussed

            const guildId = channel.guildId;
            const persona = guildId ? getGuild(guildId).persona : DEFAULT_PERSONA;
            const contextTail = history.slice(-4).map((m) => `${m.role === 'user' ? 'User' : 'Bot'}: ${m.content}`).join('\n');

            const sys =
                persona +
                '\n\nThe user went quiet mid-conversation a little while ago and hasn\'t replied since. Write a ' +
                'SHORT, warm, natural check-in message (one sentence, maybe two) — like a friend noticing someone ' +
                'trailed off, curious or a little concerned depending on what you were talking about. Don\'t be ' +
                'clingy or dramatic, and don\'t apologize for checking in. Reference what you were just discussing ' +
                'if it makes sense to.';

            let checkinText;
            try {
                const result = await getGeneralChatReply([
                    { role: 'system', content: sys },
                    { role: 'user', content: `Recent conversation:\n${contextTail}\n\n(Write the check-in message now.)` },
                ]);
                checkinText = result.text;
            } catch (e) {
                console.error('Presence check-in generation failed, using fallback line:', e.message);
                checkinText = "Hey, you still there? 👀";
            }

            await channel.send(checkinText || "Hey, you still around?");
            pushHistory(channelId, { role: 'assistant', content: checkinText || 'Hey, you still around?' });
            markCheckinSent(channelId, userId);
            console.log(`👋 Sent idle check-in in channel ${channelId} for user ${userId}`);
        } catch (e) {
            console.error('presenceSweep failed for a thread (non-fatal):', e.message);
        }
    }
}
setInterval(() => { presenceSweep().catch((e) => console.error('presenceSweep crashed:', e.message)); }, IDLE_SWEEP_INTERVAL_MS);
setInterval(() => { pollBridgeForVoiceRequests().catch((e) => console.error('bridge poll crashed:', e.message)); }, BRIDGE_POLL_INTERVAL_MS);

// ============================================================
// 13. Main handler
// ============================================================
client.once('ready', () => {
    console.log(`Bot logged in successfully as: ${client.user.tag} 🚀`);
});

client.on('messageCreate', async (message) => {
    if (message.author.bot) return;

    if (message.content.startsWith(PREFIX) && message.content.length > PREFIX.length) {
        try {
            await handleSlashLikeCommand(message);
        } catch (e) {
            console.error('Command error:', e);
            message.reply('Something went wrong running that command.').catch(() => {});
        }
        return;
    }

    const isOwnPrivateAiChannel = message.channel.topic === `private-ai-chat:${message.author.id}`;
    const repliedToBotMessage = await getRepliedToBotMessage(message);
    if (!message.mentions.has(client.user) && !isOwnPrivateAiChannel && !repliedToBotMessage) return;

    let cleanPrompt = message.content.replace(/<@!?\d+>/g, '').trim();

    if (repliedToBotMessage) {
        const priorAnswer = repliedToBotMessage.content.slice(0, 600);
        cleanPrompt = `(Replying to your previous message: "${priorAnswer}")\n${cleanPrompt}`;
    }

    const otherAttachments = collectOtherAttachments(message);
    if (otherAttachments.length) {
        const names = otherAttachments.map((a) => a.name).join(', ');
        cleanPrompt = `${cleanPrompt}\n(User also attached file(s): ${names})`.trim();
    }

    if (!cleanPrompt && message.attachments.size === 0) {
        return message.reply(`Hi! What can I help you with today? (Type \`${PREFIX}help\` for the command list)`);
    }

    try {
        await message.channel.sendTyping();

        const isAdmin = message.member?.permissions.has(PermissionsBitField.Flags.Administrator) ?? false;
        const imageUrls = await collectImageUrls(message);
        const hasImages = imageUrls.length > 0;
        const user = getUser(message.author.id, message.author.username);
        const channelId = message.channelId;
        const isCode = looksLikeCode(cleanPrompt);
        const history = getChannelHistory(channelId);

        // NEW — presence awareness: figure out if this user is returning
        // after a meaningful gap, BEFORE we touch presence for this turn.
        const { wasAway, awayForMs } = touchPresence(channelId, message.author.id);
        let presenceNote = null;
        if (wasAway) {
            const minutes = Math.round(awayForMs / 60000);
            if (awayForMs <= WELCOME_BACK_WINDOW_MS) {
                presenceNote =
                    `PRESENCE NOTE: This user went quiet for about ${minutes} minute(s) mid-conversation and has ` +
                    `just come back. Naturally acknowledge that they're back (briefly, warmly, no big deal) before ` +
                    `or while answering their message — don't ignore the gap, but don't make it awkward either.`;
            } else {
                presenceNote =
                    `PRESENCE NOTE: This user has been gone a while (roughly ${minutes} minutes) and is now back. ` +
                    `Give a genuine, warm little "welcome back" moment before getting into their message.`;
            }
        }

        // NEW — if this text message is actually the user answering something
        // the voice-chat AI asked for (a file/text the voice bot requested via
        // the bridge), forward it to voice as well so that conversation stays
        // in sync, and let the model know so its text reply can acknowledge it.
        let bridgeNote = null;
        if (dbEnabled() && message.guildId) {
            try {
                const { data: pendingVoiceAsks } = await supabase
                    .from('bot_bridge')
                    .select('*')
                    .eq('kind', 'voice_request')
                    .eq('guild_id', message.guildId)
                    .eq('user_id', message.author.id)
                    .eq('status', 'delivered')
                    .order('created_at', { ascending: false })
                    .limit(1);
                if (pendingVoiceAsks && pendingVoiceAsks.length) {
                    const ask = pendingVoiceAsks[0];
                    const summary = otherAttachments.length
                        ? `Sent file(s): ${otherAttachments.map((a) => a.name).join(', ')}${cleanPrompt ? ` — with note: "${cleanPrompt}"` : ''}`
                        : cleanPrompt;
                    await sendBridgeToVoice(message.guildId, message.author.id, summary, { fulfillsRequestId: ask.id });
                    await supabase.from('bot_bridge').update({ status: 'fulfilled' }).eq('id', ask.id);
                    bridgeNote =
                        'PRESENCE NOTE (voice bridge): This message is the user responding to something your ' +
                        'voice-chat self asked them for a moment ago. Acknowledge that you got it and that ' +
                        "you'll bring it back into the voice conversation, in your own natural words.";
                    console.log(`🔗 User ${message.author.id} fulfilled voice request ${ask.id} via text.`);
                }
            } catch (e) {
                console.error('voice-bridge fulfillment check failed (non-fatal):', e.message);
            }
        }
        if (bridgeNote) presenceNote = presenceNote ? `${presenceNote}\n\n${bridgeNote}` : bridgeNote;

        // Retrieval-augmented context: look up anything relevant before answering.
        // FIX: this used to be skipped for hasImages, which is fine, but was otherwise
        // the ONLY way facts got pulled in for non-tool-using paths (general chat) —
        // buildSystemPrompt below always also pulls the user's full fact list now,
        // so general chat / vision replies get durable memory even without going
        // through the tool-calling path.
        const [retrievedFacts, retrievedKnowledge] = hasImages
            ? [[], []]
            : await Promise.all([
                dbSearchUserFacts(message.author.id, cleanPrompt),
                dbSearchKnowledge(message.guildId, cleanPrompt),
            ]);
        const systemPrompt = await buildSystemPrompt(message.guildId, message.author.id, isAdmin, retrievedFacts, retrievedKnowledge, presenceNote);

        let botReply;
        let providerUsed;

        if (hasImages) {
            const result = await getVisionReply(systemPrompt, history, cleanPrompt, imageUrls);
            botReply = result.text;
            providerUsed = result.provider;
        } else if (isAdmin || isCode) {
            let messages = [{ role: 'system', content: systemPrompt }, ...history, { role: 'user', content: cleanPrompt }];
            const activeTools = isAdmin ? [...knowledgeTools, ...adminTools] : knowledgeTools;
            const ctx = { guild: message.guild, userId: message.author.id, guildId: message.guildId, isAdmin };

            let response = await mistral.chat.complete({
                model: TEXT_MODEL,
                messages,
                tools: activeTools,
                toolChoice: 'auto',
                maxTokens: isCode ? 1800 : 900,
            });
            let choice = response.choices[0];

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
                        toolResult = { ok: false, result: 'Only an Admin is allowed to perform this action.' };
                    } else {
                        try {
                            toolResult = await executeTool(ctx, fnName, args);
                        } catch (e) {
                            console.error(`Tool ${fnName} failed:`, e.message);
                            toolResult = { ok: false, result: `Error running "${fnName}": ${e.message}` };
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
            botReply = choice.message.content;
            if (looksGarbled(botReply)) {
                console.error('Mistral output looked garbled/mixed-script, retrying once.');
                messages.push({ role: 'user', content: '(Your previous answer had a display glitch with mixed-up characters. Please answer again, using only normal text.)' });
                const retryResp = await mistral.chat.complete({ model: TEXT_MODEL, messages, maxTokens: isCode ? 1800 : 900 });
                botReply = retryResp.choices[0].message.content || botReply;
            }
            providerUsed = `mistral/${TEXT_MODEL}`;
        } else {
            const messages = [{ role: 'system', content: systemPrompt }, ...history, { role: 'user', content: cleanPrompt }];
            const result = await getGeneralChatReply(messages);
            botReply = result.text;
            providerUsed = result.provider;
        }

        botReply = botReply || '(no response)';
        console.log(`[${message.guild?.name || 'DM'}] replied via ${providerUsed} (${botReply.length} chars)`);

        pushHistory(channelId, { role: 'user', content: hasImages ? `${cleanPrompt} [sent ${imageUrls.length} image(s)]` : cleanPrompt });
        pushHistory(channelId, { role: 'assistant', content: botReply });
        saveStoreSoon();

        user.msgCount += 1;
        saveStoreSoon();
        // Fire-and-forget: decide what (if anything) is worth remembering long-term.
        learnAndStore(message.author.id, message.guildId, isAdmin, cleanPrompt, history);

        await sendLongReply(message, botReply);
    } catch (error) {
        console.error('Execution Error:', error?.body || error?.rawValue || error);
        await message.reply('Something went wrong connecting to the AI. Please try again in a moment!').catch(() => {});
    }
});

process.on('SIGTERM', () => {
    fs.writeFileSync(DATA_FILE, JSON.stringify(store, null, 2));
    process.exit(0);
});

client.login(process.env.DISCORD_TOKEN);
