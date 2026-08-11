import base64
import hashlib
import os
import re
import tempfile
import urllib.parse
from textwrap import dedent

import streamlit as st
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.openai import OpenAIResponses
from agno.vectordb.lancedb import LanceDb, SearchType
from openai import OpenAI

st.set_page_config(page_title="Jarvis Voice Assistant", page_icon="🤖", layout="wide")

DB_FILE = "tmp/jarvis.db"
LANCEDB_URI = "tmp/jarvis_lancedb"


def init_session_state():
    defaults = {
        "messages": [],
        "processed_audio_hash": None,
        "processed_docs": set(),
        "selected_voice": "echo",
        "user_id": "default_user",
        "pending_images": [],
        "pending_videos": [],
        "pending_errors": [],
        "conversation_mode": False,
        "mic_counter": 0,
        "quick_prompt": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_saved_api_key() -> str:
    """Read the OpenAI key from Streamlit secrets or the environment."""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")


def load_knowledge(api_key: str) -> Knowledge:
    return Knowledge(
        vector_db=LanceDb(
            uri=LANCEDB_URI,
            table_name="jarvis_docs",
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small", api_key=api_key),
        ),
    )


def make_image_tool(api_key: str):
    """Build an image-generation tool bound to the user's API key."""

    def generate_image(prompt: str) -> str:
        """Create a picture from a text description. Call this whenever the user
        asks you to draw, paint, generate, design, or imagine an image.

        Args:
            prompt: A vivid, detailed description of the desired image, in English.
        """
        try:
            client = OpenAI(api_key=api_key)
            result = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                n=1,
            )
            item = result.data[0]
            # dall-e-3 returns a URL by default; fall back to base64 if given.
            if getattr(item, "url", None):
                st.session_state.pending_images.append(item.url)
            elif getattr(item, "b64_json", None):
                st.session_state.pending_images.append(base64.b64decode(item.b64_json))
            else:
                raise RuntimeError("No image data returned.")
            return "The image was generated and is now shown to the user. Briefly describe what you created."
        except Exception as e:
            st.session_state.pending_errors.append(f"🎨 Image generation failed: {e}")
            return f"Image generation failed and the error was shown to the user: {e}"

    return generate_image


def play_music(youtube_url: str, song_title: str) -> str:
    """Play a song for the user by embedding its YouTube video. To use this,
    FIRST call your web search tool to find the official YouTube watch URL for
    the requested song, then call this tool with that URL and the song's name.

    Args:
        youtube_url: A full YouTube link, e.g. https://www.youtube.com/watch?v=XXXXXXXXXXX
        song_title: The song title and artist, used as a fallback search link.
    """
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([\w-]{11})", youtube_url)
    video_id = match.group(1) if match else None
    if not video_id and re.fullmatch(r"[\w-]{11}", youtube_url.strip()):
        video_id = youtube_url.strip()
    if not video_id:
        return "I need a valid YouTube link. Search the web for the song's YouTube URL first, then try again."
    st.session_state.pending_videos.append({"id": video_id, "title": song_title})
    return "The song is now playing below for the user. Confirm what you're playing."


def build_optional_google_tools(credentials_path: str) -> list:
    """Return Google Calendar and Gmail tools if credentials are available."""
    tools = []
    if not credentials_path or not os.path.exists(credentials_path):
        return tools
    try:
        from agno.tools.google.calendar import GoogleCalendarTools

        tools.append(GoogleCalendarTools(credentials_path=credentials_path))
    except Exception as e:
        st.sidebar.warning(f"Google Calendar tools unavailable: {e}")
    try:
        from agno.tools.google.gmail import GmailTools

        tools.append(GmailTools(credentials_path=credentials_path))
    except Exception as e:
        st.sidebar.warning(f"Gmail tools unavailable: {e}")
    return tools


def load_agent(api_key: str, knowledge: Knowledge, credentials_path: str) -> Agent:
    os.makedirs("tmp", exist_ok=True)
    # OpenAI's built-in web search runs server-side, so it works reliably
    # from cloud hosts where scraping-based search tools get blocked.
    tools: list = [
        {"type": "web_search_preview"},
        make_image_tool(api_key),
        play_music,
    ]
    tools.extend(build_optional_google_tools(credentials_path))

    return Agent(
        name="Jarvis",
        model=OpenAIResponses(id="gpt-4o", api_key=api_key),
        tools=tools,
        knowledge=knowledge,
        search_knowledge=True,
        db=SqliteDb(db_file=DB_FILE),
        enable_user_memories=True,
        add_history_to_context=True,
        num_history_runs=5,
        instructions=dedent("""\
            You are Jarvis, a personal voice assistant inspired by Iron Man's AI.

            PERSONALITY:
            - Address the user respectfully but warmly, with a touch of wit and
              confidence - you are capable and a little playful, like Tony
              Stark's Jarvis.
            - Keep spoken responses concise (2-4 sentences) since they are
              converted to speech. Expand only when the user asks for detail.

            CAPABILITIES:
            1. Web search: for ANY question about current or real-world
               information (weather, news, prices, sports, schedules, facts
               you are unsure about), you MUST use your built-in web search
               tool and answer with the actual information you found.
               NEVER tell the user to check a website themselves - you do
               the looking, then give the answer directly.
            2. Image generation: when the user asks you to draw, create, or
               imagine a picture, call the generate_image tool. The image is
               shown to the user automatically - just describe it briefly.
            3. Music: when the user asks to play a song or music, FIRST use web
               search to find a real YouTube watch URL for that song (prefer an
               official audio, lyric, or "- Topic" upload, which are usually
               embeddable), then call the play_music tool with that URL AND the
               song title. The player and a backup link appear automatically -
               confirm what you're playing.
            4. Documents: search the knowledge base when the user asks about
               their uploaded files.
            5. Memory: you automatically remember user preferences and facts
               from past conversations - use them to personalize answers.
            6. Calendar & email: if Google tools are available, manage the
               user's schedule and inbox on request. If they are not
               configured, explain how to enable them instead of failing.

            IMPORTANT:
            - Always respond in the same language the user speaks.
            - Never read out raw URLs, markdown syntax, or code in voice
              responses - summarize them naturally instead.
        """),
        markdown=True,
        retries=2,
    )


def transcribe_audio(client: OpenAI, audio_file) -> str:
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=("recording.wav", audio_file.getvalue()),
    )
    return transcript.text


def synthesize_speech(client: OpenAI, text: str, voice: str) -> bytes:
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        instructions="Speak as a calm, capable personal assistant.",
    )
    return response.content


def render_media(images: list, videos: list, errors: list = None):
    """Render generated images, music players, and any tool errors."""
    for err in errors or []:
        st.error(err)
    for image in images:
        st.image(image, use_container_width=True)
    for video in videos:
        # Older history may store a bare id string; normalize to a dict.
        if isinstance(video, str):
            video = {"id": video, "title": ""}
        video_id = video["id"]
        title = video.get("title", "")
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        st.video(watch_url)
        links = f"▶️ [Open in YouTube]({watch_url})"
        if title:
            search_url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(title)
            links += f" · 🔎 [Search YouTube]({search_url})"
        st.caption(f"{links} - if the player says *unavailable*, tap a link to play it.")


def run_turn(agent: Agent, client: OpenAI, user_text: str):
    st.session_state.pending_images = []
    st.session_state.pending_videos = []
    st.session_state.pending_errors = []

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.run(user_text, user_id=st.session_state.user_id)
            answer = response.content
        st.markdown(answer)

        images = list(st.session_state.pending_images)
        videos = list(st.session_state.pending_videos)
        errors = list(st.session_state.pending_errors)
        render_media(images, videos, errors)

        with st.spinner("Generating voice..."):
            try:
                audio_bytes = synthesize_speech(
                    client, answer, st.session_state.selected_voice
                )
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            except Exception as e:
                st.warning(f"Voice generation failed: {e}")
                audio_bytes = None

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "audio": audio_bytes,
            "images": images,
            "videos": videos,
        }
    )
    st.session_state.pending_images = []
    st.session_state.pending_videos = []
    st.session_state.pending_errors = []


def main():
    init_session_state()

    with st.sidebar:
        st.title("🤖 Jarvis Settings")
        saved_key = get_saved_api_key()
        if saved_key:
            api_key = saved_key
            st.success("🔑 Using the API key saved in app secrets")
        else:
            api_key = st.text_input(
                "OpenAI API Key", type="password",
                help="Used for the LLM, speech-to-text, and text-to-speech",
            )

        voices = ["alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]
        st.session_state.selected_voice = st.selectbox(
            "Voice", voices, index=voices.index(st.session_state.selected_voice)
        )
        st.session_state.user_id = st.text_input(
            "Your name", value=st.session_state.user_id,
            help="Jarvis keeps separate long-term memories per user",
        )
        st.session_state.conversation_mode = st.toggle(
            "🔄 Conversation mode",
            value=st.session_state.conversation_mode,
            help="After Jarvis replies, the mic resets so you can talk again right away.",
        )

        st.divider()
        st.subheader("📄 Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs for Jarvis to remember", type=["pdf"], accept_multiple_files=True
        )

        st.divider()
        st.subheader("📅 Google Calendar & Gmail (optional)")
        credentials_path = st.text_input(
            "Path to Google OAuth credentials.json",
            help="Leave empty to skip. See README for setup instructions.",
        )

        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    st.title("🤖 Jarvis - Your Personal Voice Assistant")
    st.caption(
        "Talk to Jarvis with your voice. It searches the web, generates images, "
        "plays music, remembers you, and reads your documents."
    )

    if not api_key:
        st.info("👈 Enter your OpenAI API key in the sidebar to wake Jarvis up.")
        return

    client = OpenAI(api_key=api_key)
    knowledge = load_knowledge(api_key)
    agent = load_agent(api_key, knowledge, credentials_path)

    # Index newly uploaded documents
    if uploaded_files:
        for uploaded in uploaded_files:
            if uploaded.name in st.session_state.processed_docs:
                continue
            with st.spinner(f"Reading {uploaded.name}..."):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                try:
                    knowledge.add_content(path=tmp_path, name=uploaded.name)
                    st.session_state.processed_docs.add(uploaded.name)
                    st.sidebar.success(f"Learned {uploaded.name}")
                finally:
                    os.unlink(tmp_path)

    # Quick-action buttons
    st.write("**Quick actions**")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("📰 News", use_container_width=True):
        st.session_state.quick_prompt = "Summarize today's 3 biggest news headlines."
    if c2.button("🌤️ Weather", use_container_width=True):
        st.session_state.quick_prompt = "What's the weather like right now?"
    if c3.button("💱 FX rate", use_container_width=True):
        st.session_state.quick_prompt = "What's today's USD to KRW exchange rate?"
    if c4.button("🎵 Music", use_container_width=True):
        st.session_state.quick_prompt = "Play an upbeat, popular song for me."

    # Replay conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_media(message.get("images", []), message.get("videos", []))
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")

    # Handle a quick-action button press
    if st.session_state.quick_prompt:
        prompt = st.session_state.quick_prompt
        st.session_state.quick_prompt = None
        run_turn(agent, client, prompt)
        if st.session_state.conversation_mode:
            st.session_state.mic_counter += 1
            st.rerun()

    # Voice input (dynamic key lets conversation mode reset the mic)
    if st.session_state.conversation_mode:
        st.caption("🟢 Conversation mode on - just tap the mic each time, no need to clear.")
    audio_input = st.audio_input(
        "🎙️ Hold to talk to Jarvis",
        key=f"mic_{st.session_state.mic_counter}",
    )
    if audio_input is not None:
        audio_hash = hashlib.md5(audio_input.getvalue()).hexdigest()
        if audio_hash != st.session_state.processed_audio_hash:
            st.session_state.processed_audio_hash = audio_hash
            with st.spinner("Listening..."):
                user_text = transcribe_audio(client, audio_input)
            if user_text.strip():
                run_turn(agent, client, user_text)
                if st.session_state.conversation_mode:
                    st.session_state.mic_counter += 1
                    st.rerun()

    # Text input fallback
    if user_text := st.chat_input("...or type your message"):
        run_turn(agent, client, user_text)


if __name__ == "__main__":
    main()
