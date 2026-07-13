require('dotenv').config();
// >>> FIX: Node 20 doesn't have a native WebSocket global yet (only Node 22+ does).
// Kept here even though we no longer use Supabase, in case something else in the
// dependency tree still expects a global WebSocket to exist.
const WebSocket = require('ws');
global.WebSocket = WebSocket;

const { Client, GatewayIntentBits, Partials, PermissionsBitField } = require('discord.js');
const { createClient } = require('@supabase/supabase-js');
const { Mistral } = require('@mistralai/mistralai');
// >>> VOICE ADDITION: voice connection primitives from @discordjs/voice
const { joinVoiceChannel, VoiceConnectionStatus, entersState, EndBehaviorType, createAudioPlayer, createAudioResource, AudioPlayerStatus, StreamType, NoSubscriberBehavior } = require('@discordjs/voice');
// >>> VOICE ADDITION (step 5) — free neural TTS (no API key) + the ffmpeg
// binary @discordjs/voice needs to transcode the TTS output into Opus for Discord.
const { MsEdgeTTS, OUTPUT_FORMAT } = require('msedge-tts');
const ffmpegPath = require('ffmpeg-static');
process.env.FFMPEG_PATH = ffmpegPath;
// >>> VOICE ADDITION (step 2): prism-media decodes the Opus audio Discord sends
// into raw PCM so we can write it out as a playable WAV file.
const prism = require('prism-media');
const http = require('http');
const fs = require('fs');
const path = require('path');

// ============================================================
// NOTE ON THIS VERSION
// ============================================================
// This build is VOICE-ONLY. All text-chat features from the original bot
// (Mistral general/code/vision chat, admin tools, Supabase long-term
// knowledge-base facts, per-channel chat history, private AI text channels)
// have been removed. All that's left is: join a voice channel, listen,
// transcribe (Groq Whisper), generate a short spoken reply (Groq chat), speak
// it back with free Edge TTS, and save each recording + its transcript to
// Supabase instead of just deleting the WAV after transcription.
//
// THIS VERSION ALSO FIXES: the bot's speech getting cut off by background
// noise (false barge-in), and Whisper hallucinating "Thank you." / similar
// stock phrases on near-silent audio that was long enough to pass the old
// byte-length-only checks. See the SILENCE_RMS_THRESHOLD / computeRms /
// no_speech_prob sections below for details.
//
// SUPABASE SETUP FOR RECORDINGS (in addition to any earlier setup you had):
//   1. In your Supabase project's SQL editor, run:
//        create table voice_recordings (
//          id uuid primary key default gen_random_uuid(),
//          guild_id text not null,
//          user_id text not null,
//          username text,
//          transcript text,
//          audio_url text,
//          byte_length integer,
//          created_at timestamptz default now()
//        );
//   2. In Storage, create a PUBLIC bucket named "voice-recordings".
//   3. Make sure SUPABASE_URL and SUPABASE_SERVICE_KEY are set in your .env.
//      If you see "signature verification failed" in the logs, your service
//      key is invalid/expired/rotated — regenerate the "service_role" key in
//      Supabase → Project Settings → API and update .env. That's a config
//      issue, not a code bug, and recordings just silently stop persisting
//      until it's fixed (transcription/replies keep working fine).

// ============================================================
// 1. Fake web server (keeps Render's port-binding check happy)
// ============================================================
const port = process.env.PORT || 3000;
http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('Voice bot is running!\n');
}).listen(port, () => {
    console.log(`Web server listening on port ${port} to satisfy Render requirements.`);
});

// ============================================================
// 2. Discord client
// ============================================================
const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,   // only needed for the "-voice" prefix commands
        GatewayIntentBits.MessageContent,  // only needed for the "-voice" prefix commands
        GatewayIntentBits.GuildVoiceStates, // required for voice channel join/state tracking
    ],
    partials: [Partials.Message, Partials.Channel],
});

const PREFIX = '-'; // command prefix for "-voice join" / "-voice leave" etc.

// ============================================================
// 2b. Supabase — persists voice recordings + transcripts
// ============================================================
let supabase = null;
if (process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_KEY) {
    supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY);
    console.log('✅ Supabase connected — voice recordings will be saved to the database.');
} else {
    console.warn('⚠️  SUPABASE_URL / SUPABASE_SERVICE_KEY not set. Recordings will be transcribed ' +
        'but NOT saved anywhere — the WAV is deleted right after transcription, same as before.');
}
function dbEnabled() { return supabase !== null; }
const RECORDINGS_BUCKET = 'voice-recordings';

// >>> FIX: Groq's free tier has a shared DAILY token cap across ALL requests
// in the org (TPD — tokens per day), not just a per-minute rate limit. Once
// that's hit, BOTH Groq models fail for the rest of the day. Mistral has a
// completely separate free-tier quota, so it's added below as a genuine
// third option. It's only used if MISTRAL_API_KEY is set; if not, the chain
// just falls back to Groq-only like before.
const mistral = process.env.MISTRAL_API_KEY ? new Mistral({ apiKey: process.env.MISTRAL_API_KEY }) : null;
const MISTRAL_MODEL_FALLBACK = 'mistral-small-latest'; // small/cheap model is plenty for short spoken replies

// If literally every provider is out of quota/down, speak a short canned
// line instead of just staying silent.
const FALLBACK_NO_PROVIDER_REPLY = "Sorry, I'm a bit overloaded right now — give me a few minutes and try again!";

// Uploads a WAV file to Supabase Storage and inserts a row with its transcript.
// Non-fatal on failure — a save error should never crash or block the voice loop.
async function saveRecordingToDb({ guildId, userId, username, transcript, filePath, byteLength }) {
    if (!dbEnabled()) return null;
    try {
        const fileBuffer = fs.readFileSync(filePath);
        const storagePath = `${guildId}/${path.basename(filePath)}`;
        const { error: uploadError } = await supabase.storage
            .from(RECORDINGS_BUCKET)
            .upload(storagePath, fileBuffer, { contentType: 'audio/wav' });
        if (uploadError) throw uploadError;

        const { data: publicUrlData } = supabase.storage.from(RECORDINGS_BUCKET).getPublicUrl(storagePath);
        const audioUrl = publicUrlData?.publicUrl || null;

        const { error: insertError } = await supabase.from('voice_recordings').insert({
            guild_id: guildId,
            user_id: userId,
            username: username || null,
            transcript: transcript || null,
            audio_url: audioUrl,
            byte_length: byteLength || null,
        });
        if (insertError) throw insertError;

        console.log(`💾 Saved recording to database: ${storagePath}`);
        return audioUrl;
    } catch (e) {
        // >>> FIX: "signature verification failed" specifically means the
        // Supabase service key is invalid/expired/rotated — not a transient
        // error. Call it out clearly instead of a generic non-fatal log so
        // it doesn't get lost in the noise every single time a recording
        // tries (and fails) to save.
        if (/signature verification failed/i.test(e.message || '')) {
            console.error(
                'saveRecordingToDb failed (non-fatal): Supabase service key looks invalid/expired ' +
                '(JWT signature verification failed). Regenerate the "service_role" key in your Supabase ' +
                'project settings (API section) and update SUPABASE_SERVICE_KEY in your .env — this is a ' +
                'config issue, not a code bug.'
            );
        } else {
            console.error('saveRecordingToDb failed (non-fatal):', e.message);
        }
        return null;
    }
}

// ============================================================
// 3. Config — model job division (voice-only)
// ============================================================
const GROQ_MODEL_PRIMARY = 'llama-3.3-70b-versatile';
const GROQ_MODEL_FALLBACK = 'openai/gpt-oss-120b';

function looksGarbled(text) {
    return /[\u0400-\u04FF]/.test(text || '');
}

// >>> FIX: Whisper has a well-known failure mode where near-silent/low-energy
// audio gets "transcribed" as a stock phrase from its training data. The
// phrase list below is a first-line filter, but it only catches phrases
// we've already seen — see NO_SPEECH_PROB_THRESHOLD below for a more
// general fix that doesn't depend on guessing every possible hallucination.
const WHISPER_HALLUCINATION_PHRASES = new Set([
    'thank you', 'thank you.', 'thanks for watching', 'thanks for watching!',
    'thank you for watching', 'thank you for watching!', 'bye', 'bye.',
    'see you next time', 'please subscribe', 'subscribe', 'thanks', 'thanks.',
    'you', 'okay', 'okay.', 'um', 'uh', '.', '..', '...', 'the', 'i',
]);
function looksLikeWhisperHallucination(text) {
    const normalized = String(text || '').trim().toLowerCase();
    return WHISPER_HALLUCINATION_PHRASES.has(normalized);
}

// ============================================================
// 3b. Groq — OpenAI-compatible REST API, called with plain fetch() using
//     snake_case fields (max_tokens, etc).
// ============================================================
async function callGroq(messages, { model, maxTokens = 150 } = {}) {
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

async function getGeneralChatReply(messages, maxTokens = 150) {
    const chain = [
        { provider: 'groq', model: GROQ_MODEL_PRIMARY },
        { provider: 'groq', model: GROQ_MODEL_FALLBACK },
        ...(mistral ? [{ provider: 'mistral', model: MISTRAL_MODEL_FALLBACK }] : []),
    ];
    let lastErr;
    for (const step of chain) {
        try {
            if (step.provider === 'mistral') {
                const resp = await mistral.chat.complete({ model: step.model, messages, maxTokens });
                const text = resp.choices[0].message.content;
                if (looksGarbled(text)) throw new Error(`Output looked garbled/mixed-script from ${step.model}, trying next provider`);
                return { text, provider: `mistral/${step.model}` };
            }
            const data = await callGroq(messages, { model: step.model, maxTokens });
            const text = data.choices[0].message.content;
            if (looksGarbled(text)) throw new Error(`Output looked garbled/mixed-script from ${step.model}, trying next provider`);
            return { text, provider: `groq/${step.model}` };
        } catch (e) {
            console.error(`Voice-chat provider failed (${step.provider}/${step.model}):`, e.message);
            lastErr = e;
        }
    }
    throw lastErr || new Error('All voice-chat providers failed');
}

// ============================================================
// 4. VOICE — join/leave a voice channel
// ============================================================
const voiceConnections = new Map(); // guildId -> VoiceConnection

async function joinUserVoiceChannel(message) {
    const member = message.member;
    const voiceChannel = member?.voice?.channel;
    if (!voiceChannel) {
        return { ok: false, result: 'You need to be in a voice channel first!' };
    }

    const existing = voiceConnections.get(message.guildId);
    if (existing && existing.joinConfig.channelId === voiceChannel.id) {
        return { ok: true, result: `Already connected to ${voiceChannel.name}.` };
    }

    try {
        const connection = joinVoiceChannel({
            channelId: voiceChannel.id,
            guildId: message.guildId,
            adapterCreator: message.guild.voiceAdapterCreator,
            selfDeaf: false, // we need to actually hear users
        });

        connection.on('stateChange', (oldState, newState) => {
            console.log(`Voice connection state: ${oldState.status} -> ${newState.status}`);
        });
        connection.on('error', (err) => {
            console.error('Voice connection error event:', err.message);
        });

        await entersState(connection, VoiceConnectionStatus.Ready, 20_000);
        voiceConnections.set(message.guildId, connection);

        connection.on(VoiceConnectionStatus.Disconnected, () => {
            console.log(`Voice disconnected in guild ${message.guildId}`);
            voiceConnections.delete(message.guildId);
        });

        console.log(`✅ Joined voice channel "${voiceChannel.name}" in guild ${message.guildId}`);
        startListeningForSpeech(connection, message.guildId);
        return { ok: true, result: `🔊 Joined **${voiceChannel.name}**! I'm listening — just talk.` };
    } catch (e) {
        console.error('Voice join failed:', e.message, e.stack);
        return { ok: false, result: `Couldn't join voice channel: ${e.message}` };
    }
}

function leaveVoiceChannel(guildId) {
    const connection = voiceConnections.get(guildId);
    if (!connection) return { ok: false, result: "I'm not in a voice channel." };
    connection.destroy();
    voiceConnections.delete(guildId);
    return { ok: true, result: '👋 Left the voice channel.' };
}

// ============================================================
// 5. VOICE — capture speech and save as WAV
// ============================================================
const RECORDINGS_DIR = path.join(__dirname, 'recordings');
if (!fs.existsSync(RECORDINGS_DIR)) fs.mkdirSync(RECORDINGS_DIR);

try {
    for (const f of fs.readdirSync(RECORDINGS_DIR)) {
        fs.unlinkSync(path.join(RECORDINGS_DIR, f));
    }
} catch (e) {
    console.error('Could not clear old recordings on startup:', e.message);
}

const SAMPLE_RATE = 48000;
const CHANNELS = 2;
const BIT_DEPTH = 16;

const VOICE_BURST_SILENCE_MS = 1200;
const VOICE_FINALIZE_GAP_MS = 1000;

// Skip anything shorter than this — avoids transcribing coughs/mic bumps/silence.
const MIN_UTTERANCE_BYTES = 40000;

// >>> FIX: byte length alone can't tell noise from speech — a long, quiet hum
// or mic hiss can easily be "long enough" while containing no actual voice.
// RMS (root-mean-square) amplitude measures actual signal energy, so it's
// used alongside byte length everywhere a "is this real speech?" decision
// gets made. 16-bit PCM samples range roughly -32768..32767; typical room
// noise / mic hiss sits well under this threshold, real speech sits well
// over it. Tune UP if quiet-noise false positives still get through, tune
// DOWN if quiet real speech is being dropped.
const SILENCE_RMS_THRESHOLD = 300;

function computeRms(pcmBuffer) {
    if (!pcmBuffer || pcmBuffer.length < 2) return 0;
    let sumSquares = 0;
    const sampleCount = Math.floor(pcmBuffer.length / 2);
    for (let i = 0; i < sampleCount * 2; i += 2) {
        const sample = pcmBuffer.readInt16LE(i);
        sumSquares += sample * sample;
    }
    return Math.sqrt(sumSquares / sampleCount);
}

function buildWavHeader(pcmLength) {
    const header = Buffer.alloc(44);
    const byteRate = SAMPLE_RATE * CHANNELS * (BIT_DEPTH / 8);
    const blockAlign = CHANNELS * (BIT_DEPTH / 8);

    header.write('RIFF', 0);
    header.writeUInt32LE(36 + pcmLength, 4);
    header.write('WAVE', 8);
    header.write('fmt ', 12);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20);
    header.writeUInt16LE(CHANNELS, 22);
    header.writeUInt32LE(SAMPLE_RATE, 24);
    header.writeUInt32LE(byteRate, 28);
    header.writeUInt16LE(blockAlign, 32);
    header.writeUInt16LE(BIT_DEPTH, 34);
    header.write('data', 36);
    header.writeUInt32LE(pcmLength, 40);

    return header;
}

function captureUserAudioBurst(connection, userId) {
    return new Promise((resolve) => {
        const opusStream = connection.receiver.subscribe(userId, {
            end: { behavior: EndBehaviorType.AfterSilence, duration: VOICE_BURST_SILENCE_MS },
        });

        const decoder = new prism.opus.Decoder({ rate: SAMPLE_RATE, channels: CHANNELS, frameSize: 960 });
        const chunks = [];
        opusStream.pipe(decoder);

        decoder.on('data', (chunk) => chunks.push(chunk));
        decoder.on('end', () => resolve(chunks));
        decoder.on('error', (err) => {
            console.error(`Decoder error for user ${userId}:`, err.message);
            resolve(chunks);
        });
    });
}

function writeUtteranceWav(userId, chunks) {
    const pcmData = Buffer.concat(chunks);
    const wavBuffer = Buffer.concat([buildWavHeader(pcmData.length), pcmData]);
    const filename = `utterance-${userId}-${Date.now()}.wav`;
    const filePath = path.join(RECORDINGS_DIR, filename);
    fs.writeFileSync(filePath, wavBuffer);
    console.log(`🎙️  Saved recording: ${filename} (${(wavBuffer.length / 1024).toFixed(1)} KB)`);
    return { filePath, byteLength: pcmData.length };
}

// ============================================================
// 6. VOICE — speech-to-text via Groq Whisper
// ============================================================
const GROQ_WHISPER_MODEL = 'whisper-large-v3-turbo';

// >>> FIX: below this, Whisper itself is telling us (via no_speech_prob) that
// it doesn't think there's real speech in the clip — this is exactly the
// situation that produces "Thank you." style hallucinations. Checking this
// directly is more robust than only matching a hardcoded phrase list, since
// it catches hallucinations we haven't seen yet too.
const NO_SPEECH_PROB_THRESHOLD = 0.6;

async function transcribeWithGroq(filePath) {
    const fileBuffer = fs.readFileSync(filePath);
    const formData = new FormData();
    formData.append('file', new Blob([fileBuffer], { type: 'audio/wav' }), path.basename(filePath));
    formData.append('model', GROQ_WHISPER_MODEL);
    formData.append('temperature', '0');
    formData.append('prompt', 'Casual spoken conversation, in either English or Vietnamese.');
    // verbose_json gives per-segment no_speech_prob/avg_logprob instead of just
    // plain text, so we can tell "confident real transcription" apart from
    // "Whisper guessed something on near-silent audio."
    formData.append('response_format', 'verbose_json');

    const resp = await fetch('https://api.groq.com/openai/v1/audio/transcriptions', {
        method: 'POST',
        headers: { Authorization: `Bearer ${process.env.GROQ_API_KEY || ''}` },
        body: formData,
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(`Groq Whisper: ${data?.error?.message || resp.status}`);

    const text = (data.text || '').trim();
    const segments = Array.isArray(data.segments) ? data.segments : [];
    const noSpeechProb = segments.length
        ? segments.reduce((sum, s) => sum + (typeof s.no_speech_prob === 'number' ? s.no_speech_prob : 0), 0) / segments.length
        : 0;

    return { text, noSpeechProb };
}

// ============================================================
// 7. VOICE — casual "friend" persona + AI reply
// ============================================================
const VOICE_PERSONA =
    "You're chatting out loud with a friend over voice, not writing a message — so talk like it. " +
    'Keep replies SHORT (usually 1-2 sentences, occasionally a bit more if the topic calls for it) — ' +
    'nobody wants a monologue read back to them. Use contractions, casual phrasing, and a relaxed tone, ' +
    'the way a real friend talks, not a formal assistant. Avoid sounding scripted or "AI-like" — no ' +
    'bullet points, no "I\'d be happy to help", no over-explaining. ' +
    'LANGUAGE RULE: Match whichever language the person is actually speaking in THIS turn. If their ' +
    'transcribed message is in Vietnamese, reply in natural, casual spoken Vietnamese. If it is in English, ' +
    'reply in English. Judge this from their transcript each turn, not from any earlier turn or from this ' +
    "prompt's own language. Never mix languages within a single reply. " +
    "When the person is speaking English, since they are practicing English speaking (IELTS-style), " +
    "naturally keep the conversation going — ask a genuine follow-up question sometimes, react to what " +
    "they said, disagree or joke occasionally like a real friend would, rather than just answering and " +
    "stopping. When they are speaking Vietnamese, just have a normal relaxed conversation in Vietnamese. " +
    'If multiple people are talking in the same voice channel, treat it like a real group hangout — ' +
    'you can address a specific person by name if it is clear who you are responding to.';

const VOICE_HISTORY_LIMIT = 40;
const voiceHistories = new Map();

function getVoiceHistory(guildId) {
    if (!voiceHistories.has(guildId)) voiceHistories.set(guildId, []);
    return voiceHistories.get(guildId);
}
function pushVoiceHistory(guildId, entry) {
    const hist = getVoiceHistory(guildId);
    hist.push(entry);
    while (hist.length > VOICE_HISTORY_LIMIT) hist.shift();
}

const lastFallbackNoticeAt = new Map();
const FALLBACK_NOTICE_COOLDOWN_MS = 60_000;

async function generateVoiceReply(guildId, speakerName, transcript) {
    const history = getVoiceHistory(guildId);
    const messages = [
        { role: 'system', content: VOICE_PERSONA },
        ...history,
        { role: 'user', content: `${speakerName}: ${transcript}` },
    ];

    pushVoiceHistory(guildId, { role: 'user', content: `${speakerName}: ${transcript}` });

    try {
        const result = await getGeneralChatReply(messages, 150);
        pushVoiceHistory(guildId, { role: 'assistant', content: result.text });
        return result.text;
    } catch (e) {
        console.error('Voice reply generation failed:', e.message);
        const last = lastFallbackNoticeAt.get(guildId) || 0;
        if (Date.now() - last < FALLBACK_NOTICE_COOLDOWN_MS) return null;
        lastFallbackNoticeAt.set(guildId, Date.now());
        return FALLBACK_NO_PROVIDER_REPLY;
    }
}

// ============================================================
// 8. VOICE — text-to-speech playback
// ============================================================
const VOICE_TTS_NAME_EN = 'en-US-GuyNeural';
const VOICE_TTS_NAME_VI = 'vi-VN-NamMinhNeural';
const TTS_VOLUME = 0.5;
const audioPlayers = new Map();

const VIETNAMESE_CHARS_RE = /[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]/i;
function pickTtsVoice(text) {
    return VIETNAMESE_CHARS_RE.test(text) ? VOICE_TTS_NAME_VI : VOICE_TTS_NAME_EN;
}

function getAudioPlayer(connection, guildId) {
    if (audioPlayers.has(guildId)) return audioPlayers.get(guildId);
    const player = createAudioPlayer({
        behaviors: { noSubscriber: NoSubscriberBehavior.Play },
    });
    player.on('error', (err) => {
        if (err.message && err.message.includes('turn.end')) {
            console.log('(bot got interrupted mid-reply — expected, not an error)');
        } else {
            console.error('Audio player error:', err.message);
        }
    });
    connection.subscribe(player);
    audioPlayers.set(guildId, player);
    return player;
}

function stopSpeaking(guildId) {
    const player = audioPlayers.get(guildId);
    if (player && player.state.status === AudioPlayerStatus.Playing) {
        player.stop(true);
    }
}

async function speakInVoiceChannel(connection, guildId, text) {
    if (!text || !text.trim()) return;
    try {
        const voiceName = pickTtsVoice(text);
        const tts = new MsEdgeTTS();
        await tts.setMetadata(voiceName, OUTPUT_FORMAT.WEBM_24KHZ_16BIT_MONO_OPUS);
        const { audioStream } = tts.toStream(text);

        const resource = createAudioResource(audioStream, {
            inputType: StreamType.WebmOpus,
            inlineVolume: true,
        });
        resource.volume?.setVolume(TTS_VOLUME);

        const player = getAudioPlayer(connection, guildId);
        player.play(resource);
    } catch (e) {
        console.error('TTS playback failed:', e.message);
    }
}

// ============================================================
// 9. VOICE — per-user utterance tracking (multi-user safe)
// ============================================================
const pendingUtterances = new Map();
const activeCaptures = new Set();

// >>> FIX: raised from 6000 (~31ms) to 20000 (~100ms) AND now paired with an
// RMS check (see startListeningForSpeech below). Bytes alone was far too
// easy for a mic pop or breath to satisfy, which was cutting the bot's
// speech off on background noise instead of real interruptions.
const BARGE_IN_MIN_BYTES = 20000;

function utteranceKey(guildId, userId) {
    return `${guildId}:${userId}`;
}

async function finalizeUtterance(connection, guildId, userId) {
    const key = utteranceKey(guildId, userId);
    const entry = pendingUtterances.get(key);
    if (!entry) return;
    pendingUtterances.delete(key);

    const pcmData = Buffer.concat(entry.chunks);
    if (pcmData.length < MIN_UTTERANCE_BYTES) return; // too short — likely noise, skip silently
    // >>> FIX: long-but-quiet audio (hum, hiss, distant background chatter)
    // used to sail past the length check and go straight to Whisper, which
    // is exactly what was producing repeated "Thank you." hallucinations.
    if (computeRms(pcmData) < SILENCE_RMS_THRESHOLD) {
        console.log(`🔇 Skipped low-energy audio from user ${userId} (${(pcmData.length / 1024).toFixed(1)} KB, likely noise).`);
        return;
    }

    const { filePath } = writeUtteranceWav(userId, entry.chunks);

    let speakerName = `User ${userId}`;
    try {
        const user = await client.users.fetch(userId);
        speakerName = user.username;
    } catch (_) {
        // Fall back to the generic label above if the fetch fails.
    }

    try {
        const { text: transcript, noSpeechProb } = await transcribeWithGroq(filePath);
        if (!transcript) {
            console.log(`📝 Transcript (${speakerName}): (empty — likely noise only)`);
            return;
        }
        if (noSpeechProb > NO_SPEECH_PROB_THRESHOLD) {
            console.log(`📝 Transcript (${speakerName}): "${transcript}" — high no-speech probability (${noSpeechProb.toFixed(2)}), treating as noise, ignoring.`);
            return;
        }
        if (looksLikeWhisperHallucination(transcript)) {
            console.log(`📝 Transcript (${speakerName}): "${transcript}" — looks like a Whisper silence hallucination, ignoring.`);
            return;
        }
        console.log(`📝 Transcript (${speakerName}): "${transcript}"`);

        const saveDbPromise = saveRecordingToDb({
            guildId, userId, username: speakerName, transcript, filePath, byteLength: pcmData.length,
        });

        const reply = await generateVoiceReply(guildId, speakerName, transcript);
        if (reply) {
            console.log(`🤖 Voice reply: "${reply}"`);
            await speakInVoiceChannel(connection, guildId, reply);
        }

        await saveDbPromise;
    } catch (e) {
        console.error('Whisper transcription failed:', e.message);
    } finally {
        fs.unlink(filePath, (err) => {
            if (err) console.error('Failed to delete local recording copy:', err.message);
        });
    }
}

// Starts listening for ANY user in the channel to speak. Each burst of audio
// gets appended to that user's in-progress utterance; if they go quiet for
// longer than VOICE_FINALIZE_GAP_MS, the utterance is considered finished
// and gets transcribed + replied to. A short "thinking" pause won't cut them off.
function startListeningForSpeech(connection, guildId) {
    connection.receiver.speaking.on('start', (userId) => {
        const key = utteranceKey(guildId, userId);
        if (activeCaptures.has(key)) return; // already mid-burst for this user
        activeCaptures.add(key);

        const existingEntry = pendingUtterances.get(key);
        if (existingEntry?.finalizeTimer) {
            clearTimeout(existingEntry.finalizeTimer);
            existingEntry.finalizeTimer = null;
        }

        captureUserAudioBurst(connection, userId).then((chunks) => {
            activeCaptures.delete(key);
            if (!chunks.length) return;

            const pcm = Buffer.concat(chunks);
            const burstBytes = pcm.length;
            const rms = computeRms(pcm);
            // >>> FIX: "real speech" now requires enough bytes AND enough
            // actual signal energy — not just length. This is the core fix
            // for both the false barge-in and the "Thank you." spam.
            const isRealSpeech = burstBytes >= BARGE_IN_MIN_BYTES && rms >= SILENCE_RMS_THRESHOLD;

            const entry = pendingUtterances.get(key);

            // Nobody has an utterance in progress for this user yet, and this
            // burst doesn't look like real speech — drop it entirely instead
            // of starting a new "utterance" from background noise.
            if (!entry && !isRealSpeech) return;

            // Only barge-in (cut off the bot) once we know this burst is real,
            // substantial speech — not just a mic pop or ambient noise.
            if (isRealSpeech) {
                stopSpeaking(guildId);
            }

            const target = entry || { chunks: [], finalizeTimer: null };
            target.chunks.push(...chunks);
            target.finalizeTimer = setTimeout(
                () => finalizeUtterance(connection, guildId, userId),
                VOICE_FINALIZE_GAP_MS
            );
            pendingUtterances.set(key, target);
        });
    });
}

// ============================================================
// 10. "-" prefix commands (voice-only)
// ============================================================
const HELP_TEXT = [
    '**Voice commands**',
    `\`${PREFIX}voice join\` — join your current voice channel (IELTS speaking practice mode)`,
    `\`${PREFIX}voice leave\` — leave the voice channel`,
    '',
    '**General**',
    `\`${PREFIX}help\` — show this command list`,
    `\`${PREFIX}ping\` — check bot latency`,
    `\`${PREFIX}stats\` — quick stats, including whether recordings are being saved to the database`,
].join('\n');

async function handleSlashLikeCommand(message) {
    const raw = message.content.slice(PREFIX.length).trim();
    const spaceIdx = raw.indexOf(' ');
    const cmd = (spaceIdx === -1 ? raw : raw.slice(0, spaceIdx)).toLowerCase();
    const rest = (spaceIdx === -1 ? '' : raw.slice(spaceIdx + 1)).trim();

    switch (cmd) {
        case 'help': {
            return message.reply(HELP_TEXT);
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
                `- Recording database: ${dbEnabled() ? '✅ Saving recordings + transcripts to Supabase' : '❌ Not configured (recordings are transcribed but not saved)'}`
            );
        }
        case 'voice': {
            if (!message.guild) return message.reply("This command only works inside a server, not in DMs.");
            const sub = rest.trim().toLowerCase();
            if (sub === 'join') {
                const result = await joinUserVoiceChannel(message);
                return message.reply(result.result);
            }
            if (sub === 'leave') {
                const result = leaveVoiceChannel(message.guildId);
                return message.reply(result.result);
            }
            return message.reply(`Usage: \`${PREFIX}voice join\` or \`${PREFIX}voice leave\``);
        }
        default:
            return message.reply(`Unknown command \`${PREFIX}${cmd}\`. Type \`${PREFIX}help\` for the command list.`);
    }
}

// ============================================================
// 11. Main handler
// ============================================================
client.once('ready', () => {
    console.log(`Bot logged in successfully as: ${client.user.tag} 🚀`);
});

client.on('messageCreate', async (message) => {
    if (message.author.bot) return;
    if (!message.content.startsWith(PREFIX) || message.content.length <= PREFIX.length) return;

    try {
        await handleSlashLikeCommand(message);
    } catch (e) {
        console.error('Command error:', e);
        message.reply('Something went wrong running that command.').catch(() => {});
    }
});

client.login(process.env.DISCORD_TOKEN);
