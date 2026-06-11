# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

A cookbook of 100+ **independent, self-contained LLM app templates** (AI agents, multi-agent teams, RAG, MCP agents, voice agents, fine-tuning). There is no shared code, no root build system, no monorepo tooling, and no root-level test suite. Each app directory stands alone with its own README, dependencies, and entry point. Changes to one app never affect another.

The root `README.md` is the catalog: every app is linked there by category. When adding a new app, also add it to the appropriate section of the root README.

## Repository Layout

- `starter_ai_agents/` — single-file beginner agents
- `advanced_ai_agents/` — split into `single_agent_apps/`, `multi_agent_apps/` (incl. `agent_teams/`), and `autonomous_game_playing_agent_apps/`
- `rag_tutorials/` — RAG pipelines (agentic, hybrid, corrective, local, etc.)
- `mcp_ai_agents/` — agents using MCP servers
- `voice_ai_agents/` — voice/audio agents
- `advanced_llm_apps/` — chat-with-X tutorials, memory apps, fine-tuning, token/context optimization
- `ai_agent_framework_crash_course/` — numbered lesson directories (`1_starter_agent`, `2_model_agnostic_agent`, …) for Google ADK and OpenAI Agents SDK
- `generative_ui_agents/` — full-stack Next.js + Python agent apps (CopilotKit/LangGraph); these are the only npm-based projects
- `awesome_agent_skills/` — Agent Skills (SKILL.md format per the agentskills.io spec), not runnable apps; each skill has a `SKILL.md` plus optional `scripts/` and `references/`
- `docs/` — only banner images, no documentation

## Running Apps

**Python apps (the vast majority):** each has its own `requirements.txt`. Most are Streamlit apps:

```bash
cd <category>/<app_dir>
pip install -r requirements.txt
streamlit run <app_name>.py
```

Some apps use FastAPI, plain CLI scripts, or `pyproject.toml` instead — check the app's README, which is the authoritative setup guide for that app.

**Generative UI apps** (`generative_ui_agents/*`): npm projects where `npm install` also sets up the Python agent (via `postinstall`), and `npm run dev` runs the Next.js UI and Python agent concurrently. `generative-ui-starter-project/` has its own `CLAUDE.md` with detailed architecture notes.

**API keys:** apps take provider keys (OpenAI, Anthropic, Google, etc.) either from environment variables / a `.env` file (some apps ship `.env.example`) or entered directly in the Streamlit sidebar at runtime. Never hardcode keys.

**Local variants:** many apps ship a parallel local-LLM version using Ollama (e.g. `travel_agent.py` vs `local_travel_agent.py`, or `*_local_*` app directories).

## Testing and CI

There is no repo-wide test or lint command. A handful of apps have their own tests (e.g. `advanced_ai_agents/single_agent_apps/earnings_call_analyst_agent/tests/` runs with pytest); run them from within that app's directory. The only CI workflow is `.github/workflows/claude.yml` (a `@claude`-mention PR assistant) — there is no build/test gate.

Verifying a change means running the affected app itself (most need an API key to exercise fully).

## Frameworks in Use

Apps are intentionally diverse in framework choice; match the framework already used by the app you're editing rather than converting it:

- **Agno** (`agno`) — the most common agent framework here (~50 apps), typically paired with Streamlit
- **Google ADK / Gemini** (`google-adk`, `google-genai`) — second most common; the ADK crash course covers it in depth
- **OpenAI Agents SDK** (`openai-agents`) — used in the OpenAI crash course and several agents
- **LangChain / LangGraph** — mostly in RAG tutorials and generative UI agents
- **CrewAI** — a few multi-agent apps

## Conventions for New Apps

- Place the app in the matching category directory; use `snake_case` directory names (Python apps) or `kebab-case` (generative UI apps).
- Make it self-contained: a README with step-by-step setup (clone → install → get API keys → run), a `requirements.txt`, and the app code. Single-file apps are normal and preferred for simple agents.
- Add the app to the root `README.md` catalog under its category.
- Templates are meant to be forked and run in a few commands — keep dependencies minimal and pin only what's necessary.
