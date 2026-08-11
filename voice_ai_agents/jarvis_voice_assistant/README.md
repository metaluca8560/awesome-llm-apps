# 🤖 Jarvis Voice Assistant

A personal voice assistant web app inspired by Iron Man's Jarvis. Talk to it with your voice and it talks back - while searching the web, remembering you across sessions, answering questions about your documents, and (optionally) managing your Google Calendar and Gmail.

Built with [Agno](https://github.com/agno-agi/agno), Streamlit, and OpenAI.

### Features

- **🎙️ Voice conversation**: Record with your mic in the browser; Jarvis transcribes with `gpt-4o-transcribe` and replies out loud with `gpt-4o-mini-tts` (9 selectable voices). Text chat works too.
- **🔄 Conversation mode**: Toggle it on and the mic resets after every reply, so you can keep talking without clearing the previous recording.
- **🌐 Web search**: Looks up current events, weather, prices, and facts with OpenAI's built-in web search (runs server-side, so it works even on cloud hosts).
- **🎨 Image generation**: Ask Jarvis to draw or imagine anything and it generates the picture with DALL·E 3 and shows it inline.
- **🎵 Music playback**: Ask for a song and Jarvis finds it on YouTube and embeds a player right in the chat.
- **🧠 Long-term memory**: Automatically remembers your preferences and facts across sessions (stored in local SQLite, separate memories per user name).
- **📄 Chat with your documents**: Upload PDFs and Jarvis indexes them into a local LanceDB vector store for hybrid-search RAG.
- **⚡ Quick actions**: One-tap buttons for news, weather, exchange rate, and music.
- **📅 Calendar & email (optional)**: Connect Google OAuth credentials to let Jarvis read your schedule and manage Gmail.
- **🌍 Speaks your language**: Responds in whatever language you speak to it.

### How to get Started?

1. Clone the GitHub repository

```bash
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git
cd awesome-llm-apps/voice_ai_agents/jarvis_voice_assistant
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Get your OpenAI API Key

- Sign up for an [OpenAI account](https://platform.openai.com/) and obtain your API key (used for the LLM, speech-to-text, text-to-speech, and embeddings).

4. Run the Streamlit app

```bash
streamlit run jarvis_voice_assistant.py
```

5. Enter your API key in the sidebar, allow microphone access in your browser, and start talking.

### Optional: Google Calendar & Gmail

1. Create a project in the [Google Cloud Console](https://console.cloud.google.com/) and enable the **Google Calendar API** and **Gmail API**.
2. Create **OAuth client ID** credentials (Desktop app) and download the `credentials.json` file.
3. Paste the file path into the "Google OAuth credentials" field in the sidebar. The first calendar/email request opens a browser window to authorize access.

Leave the field empty to run Jarvis without these tools - everything else works normally.

### How it Works?

Each voice turn flows through four stages:

1. **Listen**: The browser mic recording is transcribed by OpenAI `gpt-4o-transcribe`.
2. **Think**: An Agno agent (GPT-4o) handles the request. It decides when to call tools - web search, image generation, music lookup, knowledge-base search over your uploaded PDFs, or Google Calendar/Gmail - and automatically extracts user memories into SQLite, carrying the last 5 turns as conversation context.
3. **Speak & show**: The reply is synthesized with `gpt-4o-mini-tts` and auto-played, while any generated image or music player is rendered inline in the chat.
4. **Remember**: Conversation history, user memories, and document embeddings persist in the local `tmp/` directory between sessions.

The dark theme is set in `.streamlit/config.toml` at the repository root (the location Streamlit Community Cloud reads).
