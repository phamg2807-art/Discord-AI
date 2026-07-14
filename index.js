const { Client, GatewayIntentBits, Partials, PermissionsBitField, ChannelType, EmbedBuilder, ActionRowBuilder, ButtonBuilder, ButtonStyle, ModalBuilder, TextInputBuilder, TextInputStyle } = require('discord.js');
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
const RETRIEVAL_LIMIT = 6; // how many DB rows to pull into context per search (this is the bot's real memory, not a bonus)
const PREFIX = '-'; // command prefix for all "-" commands
const DATA_FILE = path.join(__dirname, 'memory.json');
const DISCORD_MAX_LEN = 2000;
const CHUNK_SIZE = 1900; // leave headroom under Discord's 2000 char limit

// FIX (truncation): token ceilings raised across the board so long,
// LaTeX/math-heavy or code-heavy answers don't get cut off mid-sentence.
// Previously these were 900 (general chat/tool-calling) / 1800 (code),
// which is what made long answers look "stuck" after a couple of chunks —
// the model's own output was being truncated before it ever got to Discord's
// chunking logic.
const MAX_TOKENS_CHAT = 2500;
const MAX_TOKENS_CODE = 3500;
const MAX_TOKENS_VISION = 2000;
const MAX_CONTINUATIONS = 3; // safety cap on auto-continue-when-truncated loop

// ============================================================
// 3b. voice<->text bridge (shared via Supabase table `bot_bridge`)
// ============================================================

const BRIDGE_POLL_INTERVAL_MS = 8_000;

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
    'If the user is an Admin and asks you to manage the server (create/delete/rename channels, manage roles, ' +
    'kick/timeout), use the matching tool instead of just describing how to do it. ' +
    "If the user's request is unclear, or if learning more about their interests/work/projects would help you " +
    'answer better in the future, feel free to ask a natural follow-up question (don\'t interrogate them — at ' +
    'most one question per turn). ' +
    'IMPORTANT: You never proactively message the user out of nowhere (no "checking in", no commenting on gaps ' +
    'in conversation, no "are you still there / thinking / wandering off" messages). You only ever speak in ' +
    'direct response to something the user just sent. ' +
    '\n\nMEMORY — YOU ARE A LIBRARIAN, NOT JUST A CHATBOT: ' +
    'You have a real long-term memory made of two layers: (1) global facts about this user, true across every ' +
    'server/channel they talk to you in, and (2) this-channel memory, specific to the private conversation ' +
    'happening right here. Treat search_knowledge like walking into a library and looking something up — use it ' +
    'proactively, BEFORE answering, any time the question could possibly connect to something said before, not ' +
    'only when the user explicitly says "remember" or "recall". Prefer checking memory over guessing or asking ' +
    'the user to repeat themselves. When you learn something durable, save it with remember_fact using the right ' +
    'scope: "user" for things true about them everywhere, "channel" for things that only make sense in the ' +
    'context of this specific private chat/thread, "guild" for server-wide info (Admins only). ' +
    'When a user shares something durable and useful about themselves, remember it using the remember_fact tool. ' +
    'Before answering something that might have been saved before, ALWAYS use the search_knowledge tool first to ' +
    'check if relevant info exists — do this proactively, not only when explicitly asked to recall something. ' +
    'If the user is replying to one of your previous messages, or the message includes a note about what they\'re ' +
    'replying to, treat that as important context for what "it"/"that"/"this" refers to. ' +
    'If a message is marked as coming from your voice-chat conversation with this same user, treat it as a ' +
    'continuation of one single ongoing relationship with them — react to it the way you would if they had just ' +
    'said it out loud to you a moment ago, and if it plainly answers something you (as the voice AI) asked them ' +
    'for, acknowledge that directly instead of treating it like a cold, out-of-nowhere message. ' +
    '\n\nIMPORTANT MEMORY BOUNDARY: Global facts about a user (their personal identity/preferences/history) are ' +
    'ONLY ever visible to you inside that user\'s own private AI chat channel. In shared/public server channels, ' +
    'you have NO access to anyone\'s personal facts — you only have this server\'s shared knowledge base. Never ' +
    'imply in a public channel that you remember personal things about someone from before; if that happens, it ' +
    'means you\'re only working from what\'s visible in this conversation right now.';

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

// ------------------------------------------------------------
// Full-text search helpers (replaces old ilike keyword matching).
// ------------------------------------------------------------
function sanitizeFtsQuery(text) {
    return String(text || '').trim().slice(0, 300);
}

async function ftsSearch(table, filters, query, limit) {
    const q = sanitizeFtsQuery(query);
    if (!q) return [];
    let builder = supabase.from(table).select('*');
    for (const [col, val] of Object.entries(filters)) builder = builder.eq(col, val);
    builder = builder.textSearch('fts', q, { type: 'websearch', config: 'english' }).limit(limit);
    const { data, error } = await builder;
    if (error) {
        console.error(`ftsSearch(${table}) failed, falling back to ILIKE:`, error.message);
        return ilikeFallbackSearch(table, filters, q, limit);
    }
    return data || [];
}

async function ilikeFallbackSearch(table, filters, query, limit) {
    const keywords = String(query)
        .split(/\s+/)
        .map((w) => w.replace(/[%,()]/g, ' ').trim())
        .filter((w) => w.length > 2)
        .slice(0, 5);
    if (!keywords.length) return [];
    const searchCols = table === 'user_facts' ? ['fact'] : ['content', 'topic'];
    const orFilter = keywords
        .flatMap((k) => searchCols.map((c) => `${c}.ilike.%${k}%`))
        .join(',');
    let builder = supabase.from(table).select('*');
    for (const [col, val] of Object.entries(filters)) builder = builder.eq(col, val);
    const { data, error } = await builder.or(orFilter).limit(limit);
    if (error) { console.error(`ilikeFallbackSearch(${table}):`, error.message); return []; }
    return data || [];
}

// ---- user_facts (global, cross-channel identity/personality facts) ----
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
    return ftsSearch('user_facts', { user_id: userId }, query, limit);
}

// ---- knowledge_base (server-wide, Admin-curated) ----
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
    return ftsSearch('knowledge_base', { guild_id: guildId }, query, limit);
}

async function dbDiagnostics(guildId, userId, channelId) {
    if (!dbEnabled()) return null;
    const [ufAll, ufMine, kbAll, kbMine, cmAll, cmMine] = await Promise.all([
        supabase.from('user_facts').select('id', { count: 'exact', head: true }),
        supabase.from('user_facts').select('id', { count: 'exact', head: true }).eq('user_id', userId),
        supabase.from('knowledge_base').select('id', { count: 'exact', head: true }),
        supabase.from('knowledge_base').select('id', { count: 'exact', head: true }).eq('guild_id', guildId),
        supabase.from('channel_memory').select('id', { count: 'exact', head: true }),
        supabase.from('channel_memory').select('id', { count: 'exact', head: true }).eq('channel_id', channelId),
    ]);
    return {
        user_facts_total: ufAll.count ?? null, user_facts_total_error: ufAll.error?.message || null,
        user_facts_mine: ufMine.count ?? null, user_facts_mine_error: ufMine.error?.message || null,
        knowledge_base_total: kbAll.count ?? null, knowledge_base_total_error: kbAll.error?.message || null,
        knowledge_base_this_guild: kbMine.count ?? null, knowledge_base_this_guild_error: kbMine.error?.message || null,
        channel_memory_total: cmAll.count ?? null, channel_memory_total_error: cmAll.error?.message || null,
        channel_memory_this_channel: cmMine.count ?? null, channel_memory_this_channel_error: cmMine.error?.message || null,
        guild_id_used: guildId,
    };
}

// ---- channel_memory (per private-AI-channel "brand database") ----
async function dbAddChannelMemory(channelId, userId, guildId, topic, content) {
    if (!dbEnabled()) return null;
    const { data, error } = await supabase
        .from('channel_memory')
        .insert({ channel_id: channelId, user_id: userId, guild_id: guildId, topic: topic || null, content })
        .select()
        .single();
    if (error) { console.error('dbAddChannelMemory:', error.message); return null; }
    return data;
}
async function dbListChannelMemory(channelId, limit = FACT_LIMIT) {
    if (!dbEnabled()) return [];
    const { data, error } = await supabase
        .from('channel_memory').select('*').eq('channel_id', channelId)
        .order('created_at', { ascending: false }).limit(limit);
    if (error) { console.error('dbListChannelMemory:', error.message); return []; }
    return data || [];
}
async function dbDeleteChannelMemory(id, channelId) {
    if (!dbEnabled()) return false;
    const { error, count } = await supabase
        .from('channel_memory').delete({ count: 'exact' }).eq('id', id).eq('channel_id', channelId);
    if (error) { console.error('dbDeleteChannelMemory:', error.message); return false; }
    return (count || 0) > 0;
}
async function dbClearChannelMemory(channelId) {
    if (!dbEnabled()) return false;
    const { error } = await supabase.from('channel_memory').delete().eq('channel_id', channelId);
    if (error) { console.error('dbClearChannelMemory:', error.message); return false; }
    return true;
}
async function dbSearchChannelMemory(channelId, query, limit = RETRIEVAL_LIMIT) {
    if (!dbEnabled() || !query) return [];
    return ftsSearch('channel_memory', { channel_id: channelId }, query, limit);
}

// ---- channel_settings (per private-AI-channel configuration) ----------
// Powers the pinned settings embed: privacy, auto-learn, tool use,
// language, verbosity, persona override, visibility, and notification prefs.
const DEFAULT_CHANNEL_SETTINGS = {
    privacy_save_memory: true,
    auto_learn: true,
    allow_tools: true,
    language_lock: null,       // null (auto) | 'en' | 'vi'
    verbosity: 'normal',       // 'concise' | 'normal' | 'detailed'
    persona_override: null,
    notify_on_mention: true,
    notify_on_reply: true,
    visible_to_everyone: false, // false = fully private (only owner can even see it), true = everyone can view/read, only owner can type
};

// Small in-memory cache so we're not round-tripping to Supabase on every
// single message just to read toggle state.
const channelSettingsCache = new Map(); // channelId -> { data, fetchedAt }
const SETTINGS_CACHE_TTL_MS = 15_000;

async function dbGetChannelSettings(channelId, userId, guildId) {
    if (!dbEnabled()) return { ...DEFAULT_CHANNEL_SETTINGS, channel_id: channelId };
    const cached = channelSettingsCache.get(channelId);
    if (cached && Date.now() - cached.fetchedAt < SETTINGS_CACHE_TTL_MS) return cached.data;

    const { data, error } = await supabase
        .from('channel_settings').select('*').eq('channel_id', channelId).maybeSingle();
    if (error) { console.error('dbGetChannelSettings:', error.message); return { ...DEFAULT_CHANNEL_SETTINGS, channel_id: channelId }; }
    if (!data) {
        const { data: created, error: insertErr } = await supabase
            .from('channel_settings')
            .insert({ channel_id: channelId, user_id: userId, guild_id: guildId, ...DEFAULT_CHANNEL_SETTINGS })
            .select()
            .single();
        if (insertErr) { console.error('dbGetChannelSettings (create):', insertErr.message); return { ...DEFAULT_CHANNEL_SETTINGS, channel_id: channelId }; }
        channelSettingsCache.set(channelId, { data: created, fetchedAt: Date.now() });
        return created;
    }
    // Backfill in case older rows predate a newly-added column (e.g. visible_to_everyone).
    const merged = { ...DEFAULT_CHANNEL_SETTINGS, ...data };
    channelSettingsCache.set(channelId, { data: merged, fetchedAt: Date.now() });
    return merged;
}

async function dbUpdateChannelSettings(channelId, patch) {
    if (!dbEnabled()) return { ok: false, settings: null, error: 'db_disabled' };
    const { data, error } = await supabase
        .from('channel_settings')
        .update({ ...patch, updated_at: new Date().toISOString() })
        .eq('channel_id', channelId)
        .select()
        .single();
    if (error) {
        console.error('dbUpdateChannelSettings:', error.message);
        return { ok: false, settings: null, error: error.message };
    }
    const merged = { ...DEFAULT_CHANNEL_SETTINGS, ...data };
    channelSettingsCache.set(channelId, { data: merged, fetchedAt: Date.now() });
    return { ok: true, settings: merged, error: null };
}

async function dbSetSettingsMessageId(channelId, messageId) {
    if (!dbEnabled()) return;
    const { error } = await supabase.from('channel_settings').update({ settings_message_id: messageId }).eq('channel_id', channelId);
    if (error) console.error('dbSetSettingsMessageId:', error.message);
    const cached = channelSettingsCache.get(channelId);
    if (cached) cached.data.settings_message_id = messageId;
}

// ---- Full wipe helpers ----------------------------------------------
async function dbWipeUserEverywhere(userId) {
    if (!dbEnabled()) return { ok: false, reason: 'db_disabled' };
    const results = await Promise.all([
        supabase.from('user_facts').delete().eq('user_id', userId),
        supabase.from('channel_memory').delete().eq('user_id', userId),
    ]);
    const errors = results.map((r) => r.error).filter(Boolean);
    return { ok: errors.length === 0, errors: errors.map((e) => e.message) };
}

async function dbWipeGuildEverywhere(guildId) {
    if (!dbEnabled()) return { ok: false, reason: 'db_disabled' };
    const results = await Promise.all([
        supabase.from('user_facts').delete().eq('guild_id', guildId),
        supabase.from('knowledge_base').delete().eq('guild_id', guildId),
        supabase.from('channel_memory').delete().eq('guild_id', guildId),
    ]);
    const errors = results.map((r) => r.error).filter(Boolean);
    return { ok: errors.length === 0, errors: errors.map((e) => e.message) };
}

function clearLocalChannelHistory(channelId) {
    if (store.channels[channelId]) {
        store.channels[channelId] = [];
        saveStoreSoon();
        return true;
    }
    return false;
}
function clearLocalGuildHistory(guildIdChannels) {
    let n = 0;
    for (const channelId of guildIdChannels) {
        if (store.channels[channelId]) { store.channels[channelId] = []; n++; }
    }
    if (n) saveStoreSoon();
    return n;
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
async function learnAndStore(userId, guildId, channelId, isPrivateAiChannel, isAdmin, recentUserText, recentHistory) {
    if (!dbEnabled() || !recentUserText || !recentUserText.trim()) return;
    try {
        const contextWindow = (recentHistory || []).slice(-6)
            .map((m) => `${m.role === 'user' ? 'User' : 'Bot'}: ${m.content}`)
            .join('\n');
        const transcript = contextWindow
            ? `Recent conversation:\n${contextWindow}\n\nUser's latest message: ${recentUserText}`
            : `User's message: ${recentUserText}`;

        const channelHint = isPrivateAiChannel
            ? 'This conversation is happening in the user\'s own private AI chat channel — a good place for ' +
              'channel-specific ongoing context (e.g. a story/project being worked on together) as well as ' +
              'global facts about the person.'
            : 'This conversation is happening in a shared/public channel.';

        const resp = await mistral.chat.complete({
            model: TEXT_MODEL,
            messages: [
                {
                    role: 'system',
                    content:
                        'Read the conversation below and decide what is worth remembering LONG-TERM, sorting it into ' +
                        'the correct bucket, like a librarian filing things on the right shelf. Be reasonably generous ' +
                        '— not just things stated directly, but anything reasonably inferable: interests, job, ongoing ' +
                        'projects, goals, relationships or opinions about specific people/things, habits, circumstances. ' +
                        `${channelHint} ` +
                        'Buckets:\n' +
                        '- "user_facts": durable facts about the PERSON that are true no matter where they talk to you ' +
                        '(identity, preferences, relationships, opinions, traits, long-running goals).\n' +
                        '- "channel_memory": context that only makes sense as part of THIS specific ongoing chat/thread ' +
                        '(e.g. details of a story, project, or plan being built up over this conversation specifically) ' +
                        '— use this instead of user_facts when the info is really about "what we are doing/discussing ' +
                        'in this room" rather than a lasting trait of the person.\n' +
                        '- "guild_knowledge": info that applies to the whole SERVER, not just this person (rules, ' +
                        'events, shared context) — only used if the requester is an Admin.\n' +
                        'ONLY skip pure greetings, one-off technical questions, or messages with no real information. ' +
                        'Write each item concisely, in third person. ' +
                        'Return ONLY JSON in this exact format: ' +
                        '{"user_facts": ["..."], "channel_memory": [{"topic": "...", "content": "..."}], ' +
                        '"guild_knowledge": [{"topic": "...", "content": "..."}]}. ' +
                        'If nothing fits a bucket, return an empty array for it.',
                },
                { role: 'user', content: transcript },
            ],
            responseFormat: { type: 'json_object' },
            maxTokens: 400,
        });
        const parsed = JSON.parse(resp.choices[0].message.content);

        if (Array.isArray(parsed.user_facts) && parsed.user_facts.length) {
            const existing = await dbListUserFacts(userId, FACT_LIMIT);
            const existingLower = new Set(existing.map((f) => f.fact.toLowerCase()));
            for (const fact of parsed.user_facts) {
                if (fact && typeof fact === 'string' && !existingLower.has(fact.toLowerCase())) {
                    const saved = await dbAddUserFact(userId, guildId, fact.slice(0, 500));
                    if (saved) console.log(`🧠 [user_facts] ${userId}: ${fact}`);
                }
            }
        }
        if (isPrivateAiChannel && Array.isArray(parsed.channel_memory) && parsed.channel_memory.length) {
            const existing = await dbListChannelMemory(channelId, FACT_LIMIT);
            const existingLower = new Set(existing.map((f) => f.content.toLowerCase()));
            for (const item of parsed.channel_memory) {
                if (item && item.content && !existingLower.has(String(item.content).toLowerCase())) {
                    const saved = await dbAddChannelMemory(channelId, userId, guildId, (item.topic || '').slice(0, 200), String(item.content).slice(0, 1000));
                    if (saved) console.log(`📎 [channel_memory] ${channelId}: ${item.content}`);
                }
            }
        }
        if (isAdmin && Array.isArray(parsed.guild_knowledge) && parsed.guild_knowledge.length) {
            for (const item of parsed.guild_knowledge) {
                if (item && item.content) {
                    const saved = await dbAddKnowledge(guildId, (item.topic || '').slice(0, 200), String(item.content).slice(0, 1000), userId);
                    if (saved) console.log(`📚 [guild_knowledge] ${guildId}: ${item.content}`);
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
const LANGUAGE_INSTRUCTIONS = {
    en: 'LANGUAGE RULE: Always reply in English, regardless of what language the user writes in. If they write in ' +
        'another language, still answer in English (you may note you noticed their language, but keep the actual ' +
        'answer in English).',
    vi: 'LANGUAGE RULE: Luôn trả lời bằng tiếng Việt, bất kể người dùng viết bằng ngôn ngữ nào. Nếu họ viết bằng ' +
        'ngôn ngữ khác, vẫn trả lời bằng tiếng Việt tự nhiên, trôi chảy.',
    auto: 'LANGUAGE RULE: Default to natural, fluent English. If a user writes to you in a different language, ' +
          'reply in that same language for that exchange instead. Never mix unrelated scripts/languages into a ' +
          'single reply unless the user explicitly asks for a translation.',
};

const VERBOSITY_INSTRUCTIONS = {
    concise: 'RESPONSE LENGTH: Keep answers short and to the point — a few sentences at most unless the user is ' +
             'clearly asking for something long-form (like code or a document). Avoid padding or repeating yourself.',
    normal: 'RESPONSE LENGTH: Use your normal judgement on length — as long as it needs to be to actually help, ' +
            'no longer.',
    detailed: 'RESPONSE LENGTH: Prefer thorough, well-explained answers. Walk through your reasoning, add relevant ' +
              'context, and don\'t be afraid of a longer reply when the topic warrants it.',
};

// FIX (privacy leak): global user_facts / channel_memory are now ONLY
// fetched and injected when isPrivateAiChannel is true. Previously
// dbListUserFacts(userId, ...) ran unconditionally, meaning personal facts
// leaked into public/shared channels. Public channels now only ever see
// this server's shared knowledge_base (via the `retrieved.knowledge` block
// further down, which is unaffected by this fix since it was already
// guild-scoped, not user-scoped).
async function buildSystemPrompt(guildId, userId, channelId, isPrivateAiChannel, isAdmin, retrieved, settings) {
    const persona = (settings && settings.persona_override) ? settings.persona_override : getGuild(guildId).persona;
    let sys = persona;

    const [facts, channelMem] = await Promise.all([
        isPrivateAiChannel ? dbListUserFacts(userId, FACT_LIMIT) : Promise.resolve([]),
        isPrivateAiChannel ? dbListChannelMemory(channelId, FACT_LIMIT) : Promise.resolve([]),
    ]);
    if (facts.length) {
        sys += `\n\nGlobal library — things you already know about this user everywhere (use naturally, don't just list them out):\n- ${facts.map((f) => f.fact).join('\n- ')}`;
    }
    if (channelMem.length) {
        sys += `\n\nThis channel's own memory shelf — context specific to this private conversation:\n- ${channelMem.map((c) => `${c.topic ? c.topic + ': ' : ''}${c.content}`).join('\n- ')}`;
    }
    if (retrieved) {
        // Personal-fact search hits are also gated to the private AI channel only.
        if (isPrivateAiChannel && retrieved.facts && retrieved.facts.length) {
            sys += `\n\nRelevant items found in the global library while looking up the current question:\n- ${retrieved.facts.map((f) => f.fact).join('\n- ')}`;
        }
        if (retrieved.channelMemory && retrieved.channelMemory.length) {
            sys += `\n\nRelevant items found on this channel's memory shelf:\n- ${retrieved.channelMemory.map((c) => `${c.topic ? c.topic + ': ' : ''}${c.content}`).join('\n- ')}`;
        }
        if (retrieved.knowledge && retrieved.knowledge.length) {
            sys += `\n\nRelevant knowledge found in this server's database (only use if actually relevant, don't make things up if unsure):\n- ${retrieved.knowledge.map((k) => `${k.topic ? k.topic + ': ' : ''}${k.content}`).join('\n- ')}`;
        }
    }
    sys += isAdmin
        ? '\n\nThe person messaging you is an Admin of this server, allowed to use any admin tool.'
        : '\n\nThe person messaging you is NOT an Admin — if they ask for an admin action, explain that only an Admin can do that, and do not call the tool.';
    if (isPrivateAiChannel) {
        sys += '\n\nThis is the user\'s own private AI chat channel — their personal "brand database" room. You may ' +
               'freely save channel-specific memory here (scope "channel") in addition to global facts (scope "user").';
    } else {
        sys += '\n\nThis is a shared/public channel — you do NOT have access to anyone\'s personal global facts or ' +
               'private channel memory here, only this server\'s shared knowledge base (if any is relevant above). ' +
               'Do not reference or imply personal memory of this user in this channel.';
    }
    if (!dbEnabled()) {
        sys += '\n\nNOTE: The long-term knowledge database is currently not configured, so you have no persistent memory of this user beyond the current conversation. Do not claim to remember things from before this chat.';
    }
    sys += '\n\nMEMORY SOURCE OF TRUTH: The facts/memory lists above reflect the database RIGHT NOW. If something ' +
           'you or the user mentioned earlier in this conversation is not in those lists anymore, treat it as ' +
           'deleted — don\'t keep asserting it just because it appeared earlier in the chat transcript.';

    // Per-channel settings now actually shape the reply, not just display state.
    if (settings) {
        const langKey = settings.language_lock === 'en' ? 'en' : settings.language_lock === 'vi' ? 'vi' : 'auto';
        sys += `\n\n${LANGUAGE_INSTRUCTIONS[langKey]}`;
        const verbKey = VERBOSITY_INSTRUCTIONS[settings.verbosity] ? settings.verbosity : 'normal';
        sys += `\n\n${VERBOSITY_INSTRUCTIONS[verbKey]}`;
        if (isPrivateAiChannel && !settings.privacy_save_memory) {
            sys += '\n\nPRIVACY MODE IS ON for this channel: the user has disabled saving new long-term memory from ' +
                   'this conversation. Do not call remember_fact for scope "channel" or "user" based on this chat, ' +
                   'even if something durable comes up — you may still read/search existing memory, just don\'t add to it.';
        }
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
async function callGroq(messages, { model, maxTokens = MAX_TOKENS_CHAT } = {}) {
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

async function callOpenRouter(messages, { model, maxTokens = MAX_TOKENS_CHAT } = {}) {
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

// FIX (truncation): general chat now also auto-continues if a provider's
// response was cut off by hitting the token cap, instead of silently
// returning a partial answer.
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
                const data = await callGroq(messages, { model: step.model, maxTokens: MAX_TOKENS_CHAT });
                let text = data.choices[0].message.content;
                if (looksGarbled(text)) throw new Error(`Output looked garbled/mixed-script from ${step.model}, trying next provider`);
                let finishReason = data.choices[0].finish_reason;
                let continueGuard = 0;
                let runningMessages = messages;
                while (finishReason === 'length' && continueGuard < MAX_CONTINUATIONS) {
                    continueGuard++;
                    runningMessages = [
                        ...runningMessages,
                        { role: 'assistant', content: text },
                        { role: 'user', content: 'Continue exactly where you left off — do not repeat any earlier part of the answer.' },
                    ];
                    const contData = await callGroq(runningMessages, { model: step.model, maxTokens: MAX_TOKENS_CHAT });
                    text += contData.choices[0].message.content || '';
                    finishReason = contData.choices[0].finish_reason;
                }
                return { text, provider: `groq/${step.model}` };
            }
            const resp = await mistral.chat.complete({ model: step.model, messages, maxTokens: MAX_TOKENS_CHAT });
            let text = resp.choices[0].message.content;
            if (looksGarbled(text)) throw new Error(`Output looked garbled/mixed-script from ${step.model}, trying next provider`);
            let finishReason = resp.choices[0].finishReason;
            let continueGuard = 0;
            let runningMessages = messages;
            while (finishReason === 'length' && continueGuard < MAX_CONTINUATIONS) {
                continueGuard++;
                runningMessages = [
                    ...runningMessages,
                    { role: 'assistant', content: text },
                    { role: 'user', content: 'Continue exactly where you left off — do not repeat any earlier part of the answer.' },
                ];
                const contResp = await mistral.chat.complete({ model: step.model, messages: runningMessages, maxTokens: MAX_TOKENS_CHAT });
                text += contResp.choices[0].message.content || '';
                finishReason = contResp.choices[0].finishReason;
            }
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
            const data = await callOpenRouter(openRouterMessages, { model, maxTokens: MAX_TOKENS_VISION });
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
    const resp = await mistral.chat.complete({ model: VISION_MODEL, messages: mistralMessages, maxTokens: MAX_TOKENS_VISION });
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
                "Search the long-term knowledge database, like looking something up in a library, before answering. " +
                "Searches across: (1) this channel's own memory shelf (if in a private AI channel), (2) the user's " +
                "global facts (true everywhere, only available inside their private AI channel), and (3) this " +
                "server's shared knowledge base. ALWAYS use this before answering if there is ANY chance the answer " +
                "might depend on something discussed, saved, or mentioned before — not just when the user " +
                "explicitly asks you to recall something. Prefer this over guessing or asking the user to repeat " +
                "context they've already given you.",
            parameters: {
                type: 'object',
                properties: { query: { type: 'string', description: 'Natural-language search phrase — full-text search, so normal words/phrases work well' } },
                required: ['query'],
            },
        },
    },
    {
        type: 'function',
        function: {
            name: 'remember_fact',
            description:
                'File a durable, useful piece of information onto the correct memory shelf. Use scope "user" for ' +
                'facts about the specific person that should follow them everywhere. Use scope "channel" for context ' +
                'that only makes sense as part of THIS specific private conversation/thread (only works inside a ' +
                'private AI chat channel). Use scope "guild" for information relevant to the whole server (only ' +
                'actually saved if the requester is an Admin).',
            parameters: {
                type: 'object',
                properties: {
                    scope: { type: 'string', enum: ['user', 'channel', 'guild'] },
                    topic: { type: 'string', description: 'Short label, mainly used for channel/guild-scope facts' },
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
        // Default state: fully private. @everyone denied ViewChannel entirely.
        // The "visible_to_everyone" setting (default off) can later widen this
        // to allow read-only viewing for the whole server.
        permissionOverwrites: [
            {
                id: guild.roles.everyone.id,
                deny: [PermissionsBitField.Flags.ViewChannel, PermissionsBitField.Flags.SendMessages],
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

    if (dbEnabled()) {
        await dbGetChannelSettings(channel.id, member.id, guild.id); // creates the settings row with defaults
        await postOrRefreshSettingsEmbed(channel, member.id, guild.id);
    }

    return { channel, created: true };
}

// Applies the visible_to_everyone toggle to the actual Discord permission
// overwrites for @everyone on this channel. Only-owner-can-type is preserved
// either way; this only changes whether @everyone can view/read.
async function applyChannelVisibility(channel, visibleToEveryone) {
    try {
        const everyoneId = channel.guild.roles.everyone.id;
        if (visibleToEveryone) {
            await channel.permissionOverwrites.edit(everyoneId, {
                ViewChannel: true,
                ReadMessageHistory: true,
                SendMessages: false,
            });
        } else {
            await channel.permissionOverwrites.edit(everyoneId, {
                ViewChannel: false,
                SendMessages: false,
            });
        }
        return true;
    } catch (e) {
        console.error('applyChannelVisibility failed:', e.message);
        return false;
    }
}

// ------------------------------------------------------------
// Settings embed — dynamic + pinned. Renders current toggle state as an
// embed with button rows underneath. Re-used both for initial post and
// for in-place edits after a button click, so the message ID stays the
// same (dynamic, not a spam of new messages).
// ------------------------------------------------------------
const VERBOSITY_CYCLE = ['concise', 'normal', 'detailed'];
const LANGUAGE_CYCLE = [null, 'en', 'vi'];

function onOff(bool) { return bool ? '✅ On' : '❌ Off'; }
function languageLabel(code) {
    if (code === 'en') return '🇬🇧 English';
    if (code === 'vi') return '🇻🇳 Vietnamese';
    return '🌐 Auto-detect';
}

function buildSettingsEmbed(settings, member) {
    return new EmbedBuilder()
        .setTitle('⚙️ Private AI Channel Settings')
        .setDescription(
            `Personal configuration for ${member ? `<@${member.id}>` : 'this'}'s private AI chat channel. ` +
            `Click the buttons below to change anything — this message updates in place and stays pinned.`
        )
        .setColor(0x5865f2)
        .addFields(
            { name: '🔒 Privacy — save memory', value: onOff(settings.privacy_save_memory), inline: true },
            { name: '🧠 Auto-learn from chat', value: onOff(settings.auto_learn), inline: true },
            { name: '🛠️ Allow AI tool use', value: onOff(settings.allow_tools), inline: true },
            { name: '👀 Visible to everyone', value: onOff(settings.visible_to_everyone), inline: true },
            { name: '🌐 Language', value: languageLabel(settings.language_lock), inline: true },
            { name: '📝 Response verbosity', value: settings.verbosity.charAt(0).toUpperCase() + settings.verbosity.slice(1), inline: true },
            { name: '🎭 Persona override', value: settings.persona_override ? 'Custom (set)' : 'Server default', inline: true },
            { name: '🔔 Notify on mention', value: onOff(settings.notify_on_mention), inline: true },
            { name: '🔔 Notify on reply', value: onOff(settings.notify_on_reply), inline: true },
        )
        .setFooter({ text: `Channel ID: ${settings.channel_id}` })
        .setTimestamp();
}

function buildSettingsButtonRows(settings) {
    const row1 = new ActionRowBuilder().addComponents(
        new ButtonBuilder().setCustomId('cs:toggle:privacy_save_memory').setLabel(settings.privacy_save_memory ? 'Privacy: On' : 'Privacy: Off').setStyle(settings.privacy_save_memory ? ButtonStyle.Success : ButtonStyle.Secondary),
        new ButtonBuilder().setCustomId('cs:toggle:auto_learn').setLabel(settings.auto_learn ? 'Auto-learn: On' : 'Auto-learn: Off').setStyle(settings.auto_learn ? ButtonStyle.Success : ButtonStyle.Secondary),
        new ButtonBuilder().setCustomId('cs:toggle:allow_tools').setLabel(settings.allow_tools ? 'Tools: On' : 'Tools: Off').setStyle(settings.allow_tools ? ButtonStyle.Success : ButtonStyle.Secondary),
        new ButtonBuilder().setCustomId('cs:toggle:visible_to_everyone').setLabel(settings.visible_to_everyone ? 'Visible: Everyone' : 'Visible: Only me').setStyle(settings.visible_to_everyone ? ButtonStyle.Success : ButtonStyle.Secondary),
    );
    const row2 = new ActionRowBuilder().addComponents(
        new ButtonBuilder().setCustomId('cs:cycle:verbosity').setLabel(`Verbosity: ${settings.verbosity}`).setStyle(ButtonStyle.Primary),
        new ButtonBuilder().setCustomId('cs:cycle:language_lock').setLabel(`Language: ${languageLabel(settings.language_lock)}`).setStyle(ButtonStyle.Primary),
        new ButtonBuilder().setCustomId('cs:toggle:notify_on_mention').setLabel(settings.notify_on_mention ? 'Notify: On' : 'Notify: Off').setStyle(settings.notify_on_mention ? ButtonStyle.Success : ButtonStyle.Secondary),
    );
    const row3 = new ActionRowBuilder().addComponents(
        new ButtonBuilder().setCustomId('cs:persona:edit').setLabel('Edit persona override').setStyle(ButtonStyle.Secondary),
        new ButtonBuilder().setCustomId('cs:persona:clear').setLabel('Clear persona override').setStyle(ButtonStyle.Danger),
        new ButtonBuilder().setCustomId('cs:refresh').setLabel('🔄 Refresh').setStyle(ButtonStyle.Secondary),
        new ButtonBuilder().setCustomId('cs:reset').setLabel('↩️ Reset to defaults').setStyle(ButtonStyle.Danger),
    );
    return [row1, row2, row3];
}

async function postOrRefreshSettingsEmbed(channel, userId, guildId) {
    if (!dbEnabled()) return null;
    const settings = await dbGetChannelSettings(channel.id, userId, guildId);
    const embed = buildSettingsEmbed(settings, { id: userId });
    const components = buildSettingsButtonRows(settings);

    if (settings.settings_message_id) {
        try {
            const existingMsg = await channel.messages.fetch(settings.settings_message_id);
            await existingMsg.edit({ embeds: [embed], components });
            return existingMsg;
        } catch (e) {
            console.error('postOrRefreshSettingsEmbed: could not fetch/edit existing message, reposting:', e.message);
        }
    }

    const sent = await channel.send({ embeds: [embed], components });
    try { await sent.pin(); } catch (e) { console.error('Could not pin settings embed:', e.message); }
    await dbSetSettingsMessageId(channel.id, sent.id);
    return sent;
}

// ============================================================
// 9. Tool execution — the actual side effects (Discord.js + DB)
// ============================================================
async function executeTool(ctx, name, args) {
    switch (name) {
        case 'search_knowledge': {
            if (!dbEnabled()) return { ok: false, result: 'The knowledge database is not configured yet.' };
            const [facts, channelMem, knowledge] = await Promise.all([
                ctx.isPrivateAiChannel ? dbSearchUserFacts(ctx.userId, args.query) : Promise.resolve([]),
                ctx.isPrivateAiChannel ? dbSearchChannelMemory(ctx.channelId, args.query) : Promise.resolve([]),
                dbSearchKnowledge(ctx.guildId, args.query),
            ]);
            if (!facts.length && !channelMem.length && !knowledge.length) {
                return { ok: true, result: 'Nothing relevant found in the database.' };
            }
            const lines = [];
            if (channelMem.length) lines.push('This channel\'s memory shelf:\n' + channelMem.map((c) => `- ${c.topic ? c.topic + ': ' : ''}${c.content}`).join('\n'));
            if (facts.length) lines.push('About this user (global):\n' + facts.map((f) => `- ${f.fact}`).join('\n'));
            if (knowledge.length) lines.push('Server knowledge:\n' + knowledge.map((k) => `- ${k.topic ? k.topic + ': ' : ''}${k.content}`).join('\n'));
            return { ok: true, result: lines.join('\n\n') };
        }
        case 'remember_fact': {
            if (!dbEnabled()) return { ok: false, result: 'The knowledge database is not configured yet.' };
            if (!args.content) return { ok: false, result: 'Missing content to remember.' };
            if (ctx.settings && ctx.isPrivateAiChannel && !ctx.settings.privacy_save_memory && args.scope !== 'guild') {
                return { ok: false, result: 'Privacy mode is on for this channel — new memory is not being saved right now.' };
            }
            if (args.scope === 'guild') {
                if (!ctx.isAdmin) return { ok: false, result: 'Only an Admin can save shared server knowledge.' };
                await dbAddKnowledge(ctx.guildId, args.topic, args.content, ctx.userId);
                return { ok: true, result: `Saved shared server knowledge: "${args.content}".` };
            }
            if (args.scope === 'channel') {
                if (!ctx.isPrivateAiChannel) return { ok: false, result: 'Channel-scoped memory can only be saved inside a private AI chat channel.' };
                await dbAddChannelMemory(ctx.channelId, ctx.userId, ctx.guildId, args.topic, args.content);
                return { ok: true, result: `Saved to this channel's memory: "${args.content}".` };
            }
            // scope "user" (global personal facts) — only saved/persisted when
            // the request happens inside the user's own private AI channel, so
            // personal memory can never be written from a public channel either.
            if (!ctx.isPrivateAiChannel) {
                return { ok: false, result: 'Personal (global) memory can only be saved inside your private AI chat channel.' };
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
    '**Personal memory (AI learns about you, global — follows you everywhere, only used in your private AI channel)**',
    `\`${PREFIX}whatyouknow\` — see what the bot remembers about you globally`,
    `\`${PREFIX}remember <thing to remember>\` — manually tell the bot to remember something about you`,
    `\`${PREFIX}forget <id>\` — forget one specific thing (id comes from whatyouknow)`,
    `\`${PREFIX}forgetme\` — erase everything the bot remembers about you globally (database only — see forgetall for a full reset)`,
    `\`${PREFIX}forgetall\` — (private AI channel only) full reset: wipes your global facts, this channel's memory, AND recent conversation history in one go`,
    '',
    '**This channel\'s memory (private AI chat only — this room\'s own "brand database")**',
    `\`${PREFIX}channelmemory\` — list what's saved specifically for this private chat`,
    `\`${PREFIX}forgetchannel <id>\` — remove one entry from this channel's memory`,
    `\`${PREFIX}forgetchannelall\` — wipe this channel's memory + recent conversation history (does not touch your global facts)`,
    '',
    '**Shared server knowledge (Admin)**',
    `\`${PREFIX}know <topic> | <content>\` — save a piece of knowledge shared by the whole server`,
    `\`${PREFIX}knowledge\` — list knowledge saved for this server`,
    `\`${PREFIX}forgetknowledge <id>\` — remove a server knowledge entry`,
    '',
    '**Database tools**',
    `\`${PREFIX}dbcheck\` — see raw row counts (facts/knowledge/channel memory) to diagnose "why isn't this working"`,
    `\`${PREFIX}dbwipe me\` — (Admin) wipe YOUR data everywhere, database + local history`,
    `\`${PREFIX}dbwipe guild\` — (Admin) ⚠️ wipe ALL data for this entire server: every user's facts, this server's knowledge base, all channel memory, and all cached conversation history. Cannot be undone`,
    '',
    '**Lookup**',
    `\`${PREFIX}search <keywords>\` — full-text search across your global facts, this channel's memory (if applicable), and server knowledge`,
    '',
    '**Private AI channel**',
    `\`${PREFIX}aichat\` — create (or open) your own private chat channel with the AI. Only you can type there, and by default only you can even see it — this can be changed with the "Visible to everyone" setting in the pinned settings panel. This channel gets its own memory database in addition to your global facts. Your personal/global facts are ONLY ever used inside this channel — public channels never see them`,
    `\`${PREFIX}settings\` — re-post/jump to your pinned settings panel for this channel`,
    '',
    `You can also **mention the bot** (\`@SjpHelper\`) or **reply to one of its messages** with a question, any time.`,
    '',
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
function isPrivateAiChannelFor(message) {
    return message.channel.topic === `private-ai-chat:${message.author.id}`;
}
// Any private AI channel at all (owned by anyone) — used to decide whether
// a *visitor* (not the owner) should be allowed to passively read replies
// without triggering their own AI turn. Visitors can never type here since
// SendMessages stays denied for @everyone regardless of visibility.
function privateAiChannelOwnerId(channel) {
    const m = /^private-ai-chat:(\d+)$/.exec(channel?.topic || '');
    return m ? m[1] : null;
}

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
                `- Tracked chat-history channels: ${Object.keys(store.channels).length}`
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
            return message.reply(`Here's what I remember about you globally:\n${lines}`);
        }
        case 'remember': {
            if (!requireDb(message)) return;
            if (!rest) return message.reply(`Enter what to remember, e.g. \`${PREFIX}remember I like modding Fabric\`.`);
            await dbAddUserFact(message.author.id, message.guildId, rest);
            return message.reply(`✅ Remembered (globally): "${rest}"`);
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
            const clearedHere = isPrivateAiChannelFor(message) ? clearLocalChannelHistory(message.channelId) : false;
            return message.reply(
                '🧹 Erased everything I globally remember about you (database).' +
                (clearedHere
                    ? ' Also cleared this channel\'s recent conversation memory so I stop echoing anything from it.'
                    : ` Note: in a shared channel I may still reference the last few lines of our recent chat here until it scrolls out — that's normal short-term context, not saved memory. Use \`${PREFIX}forgetall\` inside your private AI channel for a full reset.`)
            );
        }
        case 'forgetall': {
            if (!requireDb(message)) return;
            if (!isPrivateAiChannelFor(message)) return message.reply(`This command only works inside your own private AI chat channel (use \`${PREFIX}aichat\` to get one). It fully wipes both your global facts AND this channel's memory/history in one go.`);
            const wipe = await dbWipeUserEverywhere(message.author.id);
            clearLocalChannelHistory(message.channelId);
            if (!wipe.ok) {
                return message.reply(`⚠️ Partially wiped — some database deletes failed: ${wipe.errors.join('; ')}. Local channel memory was cleared regardless.`);
            }
            return message.reply('🧹 Full reset done: erased your global facts, this channel\'s memory, and this channel\'s recent conversation history.');
        }
        case 'channelmemory': {
            if (!requireDb(message)) return;
            if (!isPrivateAiChannelFor(message)) return message.reply('This command only works inside your own private AI chat channel.');
            const rows = await dbListChannelMemory(message.channelId);
            if (!rows.length) return message.reply("This channel's memory is empty so far.");
            const lines = rows.map((c) => `\`${c.id.slice(0, 8)}\` — ${c.topic ? `**${c.topic}**: ` : ''}${c.content}`).join('\n');
            return message.reply(`📎 This channel's memory:\n${lines}`);
        }
        case 'forgetchannel': {
            if (!requireDb(message)) return;
            if (!isPrivateAiChannelFor(message)) return message.reply('This command only works inside your own private AI chat channel.');
            if (!rest) return message.reply(`Enter the id to forget (check \`${PREFIX}channelmemory\`).`);
            const rows = await dbListChannelMemory(message.channelId, 100);
            const match = rows.find((c) => c.id.startsWith(rest));
            if (!match) return message.reply("Couldn't find that in this channel's memory.");
            await dbDeleteChannelMemory(match.id, message.channelId);
            return message.reply(`🧹 Forgot from this channel: "${match.content}"`);
        }
        case 'forgetchannelall': {
            if (!requireDb(message)) return;
            if (!isPrivateAiChannelFor(message)) return message.reply('This command only works inside your own private AI chat channel.');
            await dbClearChannelMemory(message.channelId);
            clearLocalChannelHistory(message.channelId);
            return message.reply("🧹 Wiped this channel's memory and recent conversation history. Your global facts are untouched.");
        }
        case 'know': {
            if (!requireAdmin(message)) return;
            if (!requireDb(message)) return;
            if (!rest) return message.reply(`Syntax: \`${PREFIX}know <topic> | <content>\``);
            const [topic, ...contentParts] = rest.split('|');
            const content = contentParts.join('|').trim();
            if (!content) return message.reply(`Syntax: \`${PREFIX}know <topic> | <content>\` (missing a \`|\`).`);
            const savedKnowledge = await dbAddKnowledge(message.guildId, topic.trim(), content, message.author.id);
            if (!savedKnowledge) return message.reply(`⚠️ Failed to save — check the bot logs, or run \`${PREFIX}dbcheck\` to see what's going on with the database.`);
            return message.reply(`✅ Saved knowledge "${topic.trim()}" for this server. (guild_id: \`${message.guildId}\`)`);
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
            const isPrivate = isPrivateAiChannelFor(message);
            const [facts, channelMem, knowledge] = await Promise.all([
                isPrivate ? dbSearchUserFacts(message.author.id, rest, 8) : Promise.resolve([]),
                isPrivate ? dbSearchChannelMemory(message.channelId, rest, 8) : Promise.resolve([]),
                dbSearchKnowledge(message.guildId, rest, 8),
            ]);
            if (!facts.length && !channelMem.length && !knowledge.length) return message.reply('🔎 Nothing matched that search.');
            const parts = [];
            if (channelMem.length) parts.push('**This channel\'s memory:**\n' + channelMem.map((c) => `- ${c.topic ? c.topic + ': ' : ''}${c.content}`).join('\n'));
            if (facts.length) parts.push('**About you (global):**\n' + facts.map((f) => `- ${f.fact}`).join('\n'));
            if (knowledge.length) parts.push('**Server knowledge:**\n' + knowledge.map((k) => `- ${k.topic ? k.topic + ': ' : ''}${k.content}`).join('\n'));
            return message.reply(`🔎 Results for "${rest}":\n\n${parts.join('\n\n')}`);
        }
        case 'dbcheck': {
            if (!requireDb(message)) return;
            const diag = await dbDiagnostics(message.guildId, message.author.id, message.channelId);
            const lines = [
                `**user_facts** — total rows: ${diag.user_facts_total ?? `error: ${diag.user_facts_total_error}`}, yours: ${diag.user_facts_mine ?? `error: ${diag.user_facts_mine_error}`}`,
                `**knowledge_base** — total rows: ${diag.knowledge_base_total ?? `error: ${diag.knowledge_base_total_error}`}, this guild (\`${diag.guild_id_used}\`): ${diag.knowledge_base_this_guild ?? `error: ${diag.knowledge_base_this_guild_error}`}`,
                `**channel_memory** — total rows: ${diag.channel_memory_total ?? `error: ${diag.channel_memory_total_error}`}, this channel: ${diag.channel_memory_this_channel ?? `error: ${diag.channel_memory_this_channel_error}`}`,
                `**local chat history** — this channel: ${getChannelHistory(message.channelId).length} messages cached`,
            ];
            let note = '';
            if ((diag.knowledge_base_total || 0) > 0 && (diag.knowledge_base_this_guild || 0) === 0) {
                note = '\n\n⚠️ There ARE knowledge_base rows in the database, but none match this server\'s guild_id — they were probably saved from a different server (or the `fts`/full-text migration hasn\'t been run, causing search errors). Check the `guild_id` column values in Supabase against this server\'s ID.';
            } else if ((diag.knowledge_base_total || 0) === 0) {
                note = `\n\nℹ️ knowledge_base has zero rows total — nothing has been saved yet. Use \`${PREFIX}know <topic> | <content>\` to add some.`;
            }
            return message.reply(`🔧 **DB diagnostics**\n${lines.join('\n')}${note}`);
        }
        case 'dbwipe': {
            if (!requireAdmin(message)) return;
            if (!requireDb(message)) return;
            if (!rest || !['me', 'guild'].includes(rest.trim().toLowerCase())) {
                return message.reply(`Syntax: \`${PREFIX}dbwipe me\` (wipe your own data everywhere) or \`${PREFIX}dbwipe guild\` (⚠️ wipes ALL facts/knowledge/channel memory for this whole server, Admin only, cannot be undone).`);
            }
            const mode = rest.trim().toLowerCase();
            if (mode === 'me') {
                const wipe = await dbWipeUserEverywhere(message.author.id);
                if (isPrivateAiChannelFor(message)) clearLocalChannelHistory(message.channelId);
                return message.reply(wipe.ok ? '🧹 Wiped your data across the entire database.' : `⚠️ Partially wiped, errors: ${wipe.errors.join('; ')}`);
            }
            const wipe = await dbWipeGuildEverywhere(message.guildId);
            const guildChannelIds = Object.keys(store.channels).filter((cid) => message.guild?.channels.cache.get(cid) !== undefined);
            const n = clearLocalGuildHistory(guildChannelIds);
            return message.reply(
                (wipe.ok
                    ? `🧹 Wiped ALL server data: every user's facts, this server's knowledge base, and all private-channel memory for this guild.`
                    : `⚠️ Partially wiped, errors: ${wipe.errors.join('; ')}`) +
                ` Also cleared local conversation history for ${n} tracked channel(s) — so I won't keep "remembering" anything from recent chat transcripts either.`
            );
        }
        case 'aichat': {
            if (!message.guild) return message.reply("This command only works inside a server, not in DMs.");
            const { channel, created } = await createOrGetPrivateChatChannel(message.guild, message.member);
            return message.reply(
                created
                    ? `✅ Created your private AI chat channel: ${channel}. Only you can send messages there — no mention needed, every message you send is treated as talking to me. By default only you can even see it; use the pinned ⚙️ settings panel there to make it visible to everyone (read-only for others) if you want. This channel also gets its own memory database (\`${PREFIX}channelmemory\`), separate from but layered with your global facts.`
                    : `You already have a private AI chat channel: ${channel}`
            );
        }
        case 'settings': {
            if (!requireDb(message)) return;
            if (!isPrivateAiChannelFor(message)) return message.reply('This command only works inside your own private AI chat channel.');
            const sent = await postOrRefreshSettingsEmbed(message.channel, message.author.id, message.guildId);
            if (sent && sent.id !== message.id) return; // already posted/edited in place, nothing else to say
            return;
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
function splitIntoChunks(text, maxLen = CHUNK_SIZE) {
    if (text.length <= maxLen) return [text];
    const chunks = [];
    let remaining = text;
    while (remaining.length > maxLen) {
        let breakAt = remaining.lastIndexOf('\n', maxLen);
        if (breakAt < maxLen * 0.5) breakAt = remaining.lastIndexOf(' ', maxLen);
        if (breakAt < maxLen * 0.5) breakAt = maxLen; // no good break point, hard-cut
        chunks.push(remaining.slice(0, breakAt));
        remaining = remaining.slice(breakAt).replace(/^\n+/, '');
    }
    if (remaining) chunks.push(remaining);
    return chunks;
}

// Delivers a (possibly long) final reply into a message that was already
// sent as the "Thinking for {user}..." placeholder. The first chunk edits
// that placeholder in place; any overflow chunks are sent as normal
// follow-up messages in the channel, same as the old multi-chunk behavior.
async function deliverReplyViaThinkingMessage(thinkingMessage, message, text) {
    const safeText = text && text.trim() ? text : '(no response)';
    const chunks = splitIntoChunks(safeText);
    try {
        await thinkingMessage.edit(chunks[0]);
    } catch (e) {
        console.error('Failed to edit thinking message, falling back to a fresh reply:', e.message);
        try { await message.reply(chunks[0]); } catch (_) { /* give up quietly */ }
    }
    for (let i = 1; i < chunks.length; i++) {
        try {
            await message.channel.send(chunks[i]);
        } catch (e) {
            console.error(`Failed to send chunk ${i + 1}/${chunks.length}:`, e.message);
            try {
                await message.channel.send(`⚠️ (part ${i + 1} of my reply failed to send: ${e.message})`);
            } catch (_) { /* give up quietly */ }
        }
        if (i < chunks.length - 1) await new Promise((res) => setTimeout(res, 350));
    }
}

// ============================================================
// 12. Voice bridge polling
// ============================================================
setInterval(() => { pollBridgeForVoiceRequests().catch((e) => console.error('bridge poll crashed:', e.message)); }, BRIDGE_POLL_INTERVAL_MS);

// ============================================================
// 13. Main handler
// ============================================================
client.once('ready', () => {
    console.log(`Bot logged in successfully as: ${client.user.tag} 🚀`);
});

// ------------------------------------------------------------
// Settings-panel button interactions
// ------------------------------------------------------------
client.on('interactionCreate', async (interaction) => {
    if (interaction.isButton() && interaction.customId.startsWith('cs:')) {
        try {
            if (!dbEnabled()) return interaction.reply({ content: '⚠️ The knowledge database is not configured yet.', ephemeral: true });
            const ownerId = privateAiChannelOwnerId(interaction.channel);
            if (!ownerId) return interaction.reply({ content: 'This panel only works inside a private AI chat channel.', ephemeral: true });
            if (interaction.user.id !== ownerId) {
                return interaction.reply({ content: '⛔ Only the owner of this private AI channel can change its settings.', ephemeral: true });
            }

            const [, action, field] = interaction.customId.split(':');
            let settings = await dbGetChannelSettings(interaction.channelId, ownerId, interaction.guildId);
            let writeFailedMsg = null; // set if a DB write fails, so we can tell the user without crashing

            // Small helper: apply a patch, and on failure keep the LAST KNOWN GOOD
            // settings (so the panel still renders) while flagging the failure.
            async function applyPatch(patch) {
                const result = await dbUpdateChannelSettings(interaction.channelId, patch);
                if (!result.ok) {
                    writeFailedMsg =
                        `⚠️ Couldn't save that change — the database rejected the update ` +
                        `(${result.error || 'unknown error'}). This usually means the \`channel_settings\` table ` +
                        `is missing a column. Ask whoever runs the bot to check the required schema.`;
                    return settings; // keep whatever we had before, don't null out
                }
                return result.settings;
            }

            if (action === 'toggle') {
                const patch = { [field]: !settings[field] };
                settings = await applyPatch(patch);
                if (field === 'visible_to_everyone' && !writeFailedMsg) {
                    await applyChannelVisibility(interaction.channel, settings.visible_to_everyone);
                }
            } else if (action === 'cycle') {
                if (field === 'verbosity') {
                    const idx = VERBOSITY_CYCLE.indexOf(settings.verbosity);
                    const next = VERBOSITY_CYCLE[(idx + 1) % VERBOSITY_CYCLE.length];
                    settings = await applyPatch({ verbosity: next });
                } else if (field === 'language_lock') {
                    const idx = LANGUAGE_CYCLE.indexOf(settings.language_lock);
                    const next = LANGUAGE_CYCLE[(idx + 1) % LANGUAGE_CYCLE.length];
                    settings = await applyPatch({ language_lock: next });
                }
            } else if (action === 'persona' && field === 'clear') {
                settings = await applyPatch({ persona_override: null });
            } else if (action === 'persona' && field === 'edit') {
                const modal = new ModalBuilder().setCustomId('cs-persona-modal').setTitle('Persona override for this channel');
                const input = new TextInputBuilder()
                    .setCustomId('persona_text')
                    .setLabel('How should the AI act in this channel?')
                    .setStyle(TextInputStyle.Paragraph)
                    .setPlaceholder('Leave blank + submit to clear the override.')
                    .setValue(settings.persona_override || '')
                    .setRequired(false)
                    .setMaxLength(1000);
                modal.addComponents(new ActionRowBuilder().addComponents(input));
                return interaction.showModal(modal);
            } else if (action === 'reset') {
                settings = await applyPatch({ ...DEFAULT_CHANNEL_SETTINGS });
                if (!writeFailedMsg) await applyChannelVisibility(interaction.channel, DEFAULT_CHANNEL_SETTINGS.visible_to_everyone);
            } else if (action === 'refresh') {
                // no-op patch, just re-render below
            }

            const embed = buildSettingsEmbed(settings, { id: ownerId });
            const components = buildSettingsButtonRows(settings);
            await interaction.update({ embeds: [embed], components });
            if (writeFailedMsg) {
                await interaction.followUp({ content: writeFailedMsg, ephemeral: true }).catch(() => {});
            }
        } catch (e) {
            console.error('Settings button interaction failed:', e.message);
            try {
                if (interaction.deferred || interaction.replied) {
                    await interaction.followUp({ content: `⚠️ Something went wrong: ${e.message}`, ephemeral: true });
                } else {
                    await interaction.reply({ content: `⚠️ Something went wrong: ${e.message}`, ephemeral: true });
                }
            } catch (_) { /* give up quietly */ }
        }
        return;
    }

    if (interaction.isModalSubmit() && interaction.customId === 'cs-persona-modal') {
        try {
            if (!dbEnabled()) return interaction.reply({ content: '⚠️ The knowledge database is not configured yet.', ephemeral: true });
            const ownerId = privateAiChannelOwnerId(interaction.channel);
            if (!ownerId || interaction.user.id !== ownerId) {
                return interaction.reply({ content: '⛔ Only the owner of this private AI channel can change its settings.', ephemeral: true });
            }
            const text = interaction.fields.getTextInputValue('persona_text').trim();
            const result = await dbUpdateChannelSettings(interaction.channelId, { persona_override: text || null });
            if (!result.ok) {
                return interaction.reply({ content: `⚠️ Couldn't save — the database rejected the update (${result.error || 'unknown error'}).`, ephemeral: true });
            }
            const settings = result.settings;
            const embed = buildSettingsEmbed(settings, { id: ownerId });
            const components = buildSettingsButtonRows(settings);
            const settingsMsg = settings.settings_message_id
                ? await interaction.channel.messages.fetch(settings.settings_message_id).catch(() => null)
                : null;
            if (settingsMsg) await settingsMsg.edit({ embeds: [embed], components });
            await interaction.reply({ content: text ? '✅ Persona override updated.' : '✅ Persona override cleared.', ephemeral: true });
        } catch (e) {
            console.error('Persona modal submit failed:', e.message);
            try { await interaction.reply({ content: `⚠️ Something went wrong: ${e.message}`, ephemeral: true }); } catch (_) {}
        }
    }
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

    const isPrivateAiChannel = isPrivateAiChannelFor(message);

    // Visitors browsing someone else's now-visible-to-everyone private AI
    // channel: they can never type there (SendMessages stays denied for
    // @everyone), so in practice this branch won't fire for them — but guard
    // anyway in case permissions get out of sync, so the AI never answers a
    // stranger inside someone else's private memory space.
    const ownerIdOfThisChannel = privateAiChannelOwnerId(message.channel);
    if (ownerIdOfThisChannel && ownerIdOfThisChannel !== message.author.id) return;

    const repliedToBotMessage = await getRepliedToBotMessage(message);
    if (!message.mentions.has(client.user) && !isPrivateAiChannel && !repliedToBotMessage) return;

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

    // "Thinking for {user}..." placeholder — sent immediately, edited in
    // place once the real answer is ready. Gives instant feedback instead
    // of a silent typing indicator, and doubles as visible progress if
    // something goes wrong (the edit/catch below still updates it).
    let thinkingMessage = null;
    try {
        thinkingMessage = await message.reply(`🤔 Thinking for **${message.member?.displayName || message.author.username}**...`);
    } catch (e) {
        console.error('Could not send thinking placeholder (non-fatal):', e.message);
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

        const settings = isPrivateAiChannel
            ? await dbGetChannelSettings(channelId, message.author.id, message.guildId)
            : { ...DEFAULT_CHANNEL_SETTINGS };

        // Auto-learn toggle: if the owner turned this off for their private
        // channel, skip the whole learnAndStore step below for this turn.
        const autoLearnEnabled = !isPrivateAiChannel || settings.auto_learn;

        // Allow-tools toggle: if off, general users in their own private
        // channel get plain chat instead of tool-calling for this turn.
        const toolsAllowed = !isPrivateAiChannel || settings.allow_tools;

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

        // Retrieval-augmented context: search the whole library before answering.
        // FIX (privacy leak): dbSearchUserFacts (personal global facts) is now
        // ONLY queried when isPrivateAiChannel is true. Public/shared channels
        // only ever pull dbSearchKnowledge (server-wide, guild-scoped) here.
        let retrieved = { facts: [], channelMemory: [], knowledge: [] };
        if (!hasImages || cleanPrompt) {
            const [facts, channelMem, knowledge] = await Promise.all([
                isPrivateAiChannel ? dbSearchUserFacts(message.author.id, cleanPrompt) : Promise.resolve([]),
                isPrivateAiChannel ? dbSearchChannelMemory(channelId, cleanPrompt) : Promise.resolve([]),
                dbSearchKnowledge(message.guildId, cleanPrompt),
            ]);
            retrieved = { facts, channelMemory: channelMem, knowledge };
        }
        let systemPrompt = await buildSystemPrompt(message.guildId, message.author.id, channelId, isPrivateAiChannel, isAdmin, retrieved, settings);
        if (bridgeNote) systemPrompt += `\n\n${bridgeNote}`;

        let botReply;
        let providerUsed;

        if (hasImages) {
            const result = await getVisionReply(systemPrompt, history, cleanPrompt, imageUrls);
            botReply = result.text;
            providerUsed = result.provider;
        } else if (toolsAllowed && (isAdmin || isCode || dbEnabled())) {
            let messages = [{ role: 'system', content: systemPrompt }, ...history, { role: 'user', content: cleanPrompt }];
            const activeTools = isAdmin ? [...knowledgeTools, ...adminTools] : knowledgeTools;
            const ctx = { guild: message.guild, userId: message.author.id, guildId: message.guildId, channelId, isPrivateAiChannel, isAdmin, settings };

            // FIX (truncation): maxTokens raised from 900/1800 to MAX_TOKENS_CHAT/MAX_TOKENS_CODE.
            let response = await mistral.chat.complete({
                model: TEXT_MODEL,
                messages,
                tools: activeTools,
                toolChoice: 'auto',
                maxTokens: isCode ? MAX_TOKENS_CODE : MAX_TOKENS_CHAT,
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
                response = await mistral.chat.complete({ model: TEXT_MODEL, messages, tools: activeTools, toolChoice: 'auto', maxTokens: MAX_TOKENS_CHAT });
                choice = response.choices[0];
            }
            botReply = choice.message.content;

            // FIX (truncation): if the final answer was cut off purely because
            // it hit the token cap (finishReason === 'length'), automatically
            // ask the model to continue and stitch the pieces together, instead
            // of returning (and then Discord-chunking) a partial answer.
            let continueGuard = 0;
            while (response.choices[0].finishReason === 'length' && continueGuard < MAX_CONTINUATIONS) {
                continueGuard++;
                messages.push({ role: 'assistant', content: botReply });
                messages.push({ role: 'user', content: 'Continue exactly where you left off — do not repeat any earlier part of the answer.' });
                const contResp = await mistral.chat.complete({ model: TEXT_MODEL, messages, maxTokens: MAX_TOKENS_CHAT });
                const contText = contResp.choices[0].message.content || '';
                botReply += contText;
                response = contResp;
            }

            if (looksGarbled(botReply)) {
                console.error('Mistral output looked garbled/mixed-script, retrying once.');
                messages.push({ role: 'user', content: '(Your previous answer had a display glitch with mixed-up characters. Please answer again, using only normal text.)' });
                const retryResp = await mistral.chat.complete({ model: TEXT_MODEL, messages, maxTokens: isCode ? MAX_TOKENS_CODE : MAX_TOKENS_CHAT });
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

        if (autoLearnEnabled) {
            // Fire-and-forget: decide what (if anything) is worth remembering
            // long-term, filed onto the correct shelf. Skipped entirely if the
            // channel owner has turned auto-learn off for this private channel.
            learnAndStore(message.author.id, message.guildId, channelId, isPrivateAiChannel, isAdmin, cleanPrompt, history);
        }

        if (thinkingMessage) {
            await deliverReplyViaThinkingMessage(thinkingMessage, message, botReply);
        } else {
            // Placeholder failed to send earlier (rare) — fall back to a normal reply chain.
            const chunks = splitIntoChunks(botReply && botReply.trim() ? botReply : '(no response)');
            for (let i = 0; i < chunks.length; i++) {
                if (i === 0) await message.reply(chunks[i]);
                else await message.channel.send(chunks[i]);
                if (i < chunks.length - 1) await new Promise((res) => setTimeout(res, 350));
            }
        }
    } catch (error) {
        console.error('Execution Error:', error?.body || error?.rawValue || error);
        const errText = 'Something went wrong connecting to the AI. Please try again in a moment!';
        if (thinkingMessage) {
            await thinkingMessage.edit(errText).catch(() => {});
        } else {
            await message.reply(errText).catch(() => {});
        }
    }
});

process.on('SIGTERM', () => {
    fs.writeFileSync(DATA_FILE, JSON.stringify(store, null, 2));
    process.exit(0);
});

client.login(process.env.DISCORD_TOKEN);
