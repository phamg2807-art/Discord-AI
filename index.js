const { Client, GatewayIntentBits } = require('discord.js');
const { Mistral } = require('@mistralai/mistralai');
const http = require('http'); // 1. Import built-in HTTP module

// 2. Create a fake web server to satisfy Render's port binding rule
const port = process.env.PORT || 3000;
http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('Bot is running securely!\n');
}).listen(port, () => {
    console.log(`Web server listening on port ${port} to satisfy Render requirements.`);
});

// 3. Configure the Discord client
const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent
    ]
});

// 4. Instantiate the Mistral Client
const mistral = new Mistral({ 
    apiKey: process.env.MISTRAL_API_KEY || "bP9W1oC9M6FPIw6hbGHW3jywIS16uOVd" 
});

client.once('ready', () => {
    console.log(`Bot logged in successfully as: ${client.user.tag} 🚀`);
});

client.on('messageCreate', async (message) => {
    if (message.author.bot) return;
    if (!message.mentions.has(client.user)) return;

    try {
        const cleanPrompt = message.content.replace(/<@!\d+>|<@\d+>/g, '').trim();

        if (!cleanPrompt) {
            return message.reply("Xin chào! Tôi có thể giúp gì cho bạn hôm nay?");
        }

        await message.channel.sendTyping();

        const response = await mistral.chat.complete({
            model: "mistral-large-latest", 
            messages: [
                { 
                    role: "system", 
                    content: "Bạn là một trợ lý AI thông minh, ngắn gọn, và thân thiện." 
                },
                { 
                    role: "user", 
                    content: cleanPrompt 
                }
            ]
        });

        const botReply = response.choices[0].message.content;
        
        if (botReply.length > 2000) {
            await message.reply(botReply.substring(0, 1999));
        } else {
            await message.reply(botReply);
        }

    } catch (error) {
        console.error("Execution Error:", error);
        await message.reply("Đã có lỗi xảy ra khi kết nối tới Mistral AI. Hãy thử lại sau nhé!");
    }
});

client.login(process.env.DISCORD_TOKEN);
