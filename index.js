1. Import Packages

- discord.js
- groq-sdk
- @supabase/supabase-js
- http

────────────────────────────

2. Render Web Server

Keeps Render service alive.

────────────────────────────

3. Discord Client

Login
Ready Event
Message Event

────────────────────────────

4. Groq Client

Single model

MODEL
↓

llama-3.3-70b-versatile

────────────────────────────

5. Supabase

Tables

users
messages
memories

────────────────────────────

6. Character Prompt

Kenkai's cute little sister.

────────────────────────────

7. Memory System

Load memories

↓

Load recent messages

↓

Build prompt

↓

Groq

↓

Reply

↓

Store message

↓

Extract important memory

↓

Save to Supabase

────────────────────────────

8. Reply Trigger

Reply when

• Mentioned

OR

• Inside selected channel

OR

• Replying to bot

────────────────────────────

9. Typing Indicator

Bot starts typing

↓

Generate response

↓

Reply

────────────────────────────

10. Error Handler

API errors

Discord errors

Groq errors

Supabase errors

Reconnect automatically
