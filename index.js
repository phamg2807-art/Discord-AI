const { Client, GatewayIntentBits } = require('discord.js');
const { Mistral } = require('@mistralai/mistralai');

// 1. Configure the Discord client with necessary permissions
const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent
    ]
});

// 2. Instantiate the Mistral Client (Credentials pull directly from Render settings)
const mistral = new Mistral({ 
    apiKey: process.env.MISTRAL_API_KEY || "bP9W1oC9M6FPIw6hbGHW3jywIS16uOVd" 
});

client.once('ready', () => {
    console.log(`Bot logged in successfully as: ${client.user.tag} 🚀`);
});

client.on('messageCreate', async (message) => {
    // Drop the request if it comes from another bot
    if (message.author.bot) return;

    // Check if the bot was explicitly tagged in the message
    if (!message.mentions.has(client.user)) return;

    try {
        // Strip out the Discord user ping syntax so it doesn't taint the text query
        const cleanPrompt = message.content.replace(/<@!\d+>|<@\d+>/g, '').trim();

        // If someone just pings the bot with no text, give them a gentle greeting
        if (!cleanPrompt) {
            return message.reply("Xin chào! Tôi có thể giúp gì cho bạn hôm nay?");
        }

        // Trigger the visual "is typing..." animation in the Discord channel
        await message.channel.sendTyping();

        // 3. Dispatch content to the Mistral model API
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

        // 4. Handle response delivery
        const botReply = response.choices[0].message.content;
        
        // Prevent crashes by checking Discord's strict 2000 character limits
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

// 5. Connect bot to Discord network
client.login(process.env.DISCORD_TOKEN);
