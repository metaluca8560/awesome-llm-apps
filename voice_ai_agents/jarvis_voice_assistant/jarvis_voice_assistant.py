import hashlib
import os
import tempfile
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_knowledge(api_key: str) -> Knowledge:
    return Knowledge(
        vector_db=LanceDb(
            uri=LANCEDB_URI,
            table_name="jarvis_docs",
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small", api_key=api_key),
        ),
    )


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
    tools: list = [{"type": "web_search_preview"}]
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
            - Address the user respectfully but warmly, with a touch of wit.
            - Keep spoken responses concise (2-4 sentences) since they are
              converted to speech. Expand only when the user asks for detail.

            CAPABILITIES:
            1. Web search: for ANY question about current or real-world
               information (weather, news, prices, sports, schedules, facts
               you are unsure about), you MUST use your built-in web search
               tool and answer with the actual information you found.
               NEVER tell the user to check a website themselves - you do
               the looking, then give the answer directly.
            2. Documents: search the knowledge base when the user asks about
               their uploaded files.
            3. Memory: you automatically remember user preferences and facts
               from past conversations - use them to personalize answers.
            4. Calendar & email: if Google tools are available, manage the
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


def run_turn(agent: Agent, client: OpenAI, user_text: str):
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.run(user_text, user_id=st.session_state.user_id)
            answer = response.content
        st.markdown(answer)
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
        {"role": "assistant", "content": answer, "audio": audio_bytes}
    )


def main():
    init_session_state()

    with st.sidebar:
        st.title("🤖 Jarvis Settings")
        api_key = st.text_input(
            "OpenAI API Key", type="password", help="Used for the LLM, speech-to-text, and text-to-speech"
        )

        voices = ["alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]
        st.session_state.selected_voice = st.selectbox(
            "Voice", voices, index=voices.index(st.session_state.selected_voice)
        )
        st.session_state.user_id = st.text_input(
            "Your name", value=st.session_state.user_id,
            help="Jarvis keeps separate long-term memories per user",
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
        "Talk to Jarvis with your voice. It searches the web, remembers you, "
        "reads your documents, and can manage your calendar and email."
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

    # Replay conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")

    # Voice input
    audio_input = st.audio_input("🎙️ Hold to talk to Jarvis")
    if audio_input is not None:
        audio_hash = hashlib.md5(audio_input.getvalue()).hexdigest()
        if audio_hash != st.session_state.processed_audio_hash:
            st.session_state.processed_audio_hash = audio_hash
            with st.spinner("Listening..."):
                user_text = transcribe_audio(client, audio_input)
            if user_text.strip():
                run_turn(agent, client, user_text)

    # Text input fallback
    if user_text := st.chat_input("...or type your message"):
        run_turn(agent, client, user_text)


if __name__ == "__main__":
    main()
