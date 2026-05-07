# LangGraph Blog Series

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

Companion source code for the **LangGraph** blog series published at [shafiqulai.github.io](https://shafiqulai.github.io).

All implementations use **Python 3.12**, **LangGraph**, and **Google Gemini** as the LLM. Every project follows an OOP structure — no standalone scripts.

---

## Series Overview

### Series 1 — LangGraph Basics

LangGraph applications — no matter how complex — are built from three primitives: **State**, **Nodes**, and **Edges**. Before writing a chatbot, a RAG pipeline, or a multi-agent system, those primitives need to be solid. This series covers each one in depth, starting from the simplest possible graph and adding one concept at a time until the full LangGraph toolkit is in place.

| # | Folder | Title | Blog |
|---|--------|-------|------|
| Part 1 | `basics-1-stategraph-nodes-edges/` | StateGraph, Nodes & Edges | [Read →](https://shafiqulai.github.io/blogs/blog_8.html) |
| Part 2 | `basics-2-state-annotated-reducers/` | State, Annotated Fields & Custom Reducers | [Read →](https://shafiqulai.github.io/blogs/blog_9.html) |
| Part 3 | `basics-3-conditional-edges/` | Conditional Edges & Routing Logic | Coming soon |
| Part 4 | `basics-4-checkpointers-memory/` | Checkpointers, Memory & Streaming | Coming soon |
| Part 5 | `basics-5-tools-toolnode/` | Tools, ToolNode & Prebuilt Components | Coming soon |
| Part 6 | `basics-6-subgraphs-hitl/` | Subgraphs, Interrupt & Human-in-the-Loop | Coming soon |

#### Part 1 — StateGraph, Nodes & Edges

Every LangGraph application is a graph. This part introduces the three components that every graph is built from — **State**, **Nodes**, and **Edges** — and uses them to build a simple Q&A bot powered by Google Gemini. The project includes a console runner and a Gradio web UI. The graph architecture is exported as a Mermaid diagram on every run.

| File | Responsibility |
|------|---------------|
| `config.py` | Loads `.env`, exposes `Config` class with model settings |
| `llm.py` | Wraps `ChatGoogleGenerativeAI` using `Config` |
| `state.py` | Defines `QAState` — the graph's shared state |
| `nodes.py` | Defines `QANodes.answer_node` — calls Gemini, returns partial update |
| `graph.py` | Builds and compiles the `StateGraph`, exposes it via `get_compiled_graph()` |
| `qa_runner.py` | `QARunner` — console entry point, runs three demo questions |
| `app.py` | `QAApp` — Gradio `ChatInterface` wrapping the same runner |

For a detailed explanation of every concept and code walkthrough → [https://shafiqulai.github.io/blogs/blog_8.html](https://shafiqulai.github.io/blogs/blog_8.html)

#### Part 2 — State, Annotated Fields & Custom Reducers

By default, LangGraph uses **last-write-wins**: when a node returns a value for a field, it replaces whatever was there. For fields that should *accumulate* across nodes, this silently discards data. This part introduces `Annotated` type hints with reducer functions — the mechanism that tells LangGraph to merge values instead of replacing them. Covers `operator.add`, `add_messages`, custom reducers, and the `MessagesState` shortcut. A **Topic Expander** graph (two nodes, both writing to the same `points` field) demonstrates the pattern in practice.

| File | Responsibility |
|------|---------------|
| `config.py` | Loads `.env`, exposes `Config` class with model settings |
| `llm.py` | Wraps `ChatGoogleGenerativeAI` using `Config` |
| `state.py` | Defines `TopicState` — includes `points: Annotated[list[str], operator.add]` |
| `nodes.py` | `TopicNodes` with `expand_node` and `refine_node` — loads prompts from files |
| `prompts/expand_node.txt` | Prompt template for `expand_node` (uses `{topic}` placeholder) |
| `prompts/refine_node.txt` | Prompt template for `refine_node` (uses `{topic}` and `{existing}` placeholders) |
| `graph.py` | Builds and compiles the `StateGraph`: `START → expand → refine → END` |
| `topic_runner.py` | `TopicRunner` — console entry point, runs two demo topics |
| `app.py` | `TopicApp` — Gradio `ChatInterface` wrapping the runner |

For a detailed explanation of every concept and code walkthrough → [https://shafiqulai.github.io/blogs/blog_9.html](https://shafiqulai.github.io/blogs/blog_9.html)

---

### Series 2 — LangGraph: From Zero to Production

With the basics in place, this series builds five real-world applications end to end — each one introducing a new pattern on top of the previous. The progression goes from a stateful chatbot to a multi-agent system with tool use and MCP integration.

| # | Folder | Title | Blog |
|---|--------|-------|------|
| Part 1 | `production-1-chatbot/` | Conversational Chatbot with Memory | Coming soon |
| Part 2 | `production-2-rag/` | RAG Pipeline with Conditional Routing | Coming soon |
| Part 3 | `production-3-react-agent/` | ReAct Agent with Tool Calling | Coming soon |
| Part 4 | `production-4-multi-agent/` | Multi-Agent Systems with Supervisor | Coming soon |
| Part 5 | `production-5-mcp/` | MCP Integration with LangGraph | Coming soon |

---

## Prerequisites

- Python 3.12
- A Google Gemini API key — get one free at [Google AI Studio](https://aistudio.google.com/app/apikey)

---

## Setup

All projects in this repo share a single virtual environment and `requirements.txt`.

**1. Create and activate a virtual environment:**

```bash
# Create
python -m venv langgraph

# Activate — macOS / Linux
source langgraph/bin/activate

# Activate — Windows
langgraph\Scripts\activate
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

```
langchain==1.2.17
langgraph==1.1.10
langchain-google-genai==4.2.2
python-dotenv==1.2.2
gradio==6.14.0
```

**3. Configure environment variables:**

Create a `.env` file in this folder (`langgraph/.env`) with the following:

```properties
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL_NAME=gemini-2.0-flash
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_RETRIES=2
```

> ⚠️ Never commit the `.env` file. It is already listed in `.gitignore`.

---

## Project Structure

```
langgraph/
├── basics-1-stategraph-nodes-edges/        # Series 1, Part 1
│   ├── config.py                            # Loads .env, defines Config
│   ├── llm.py                               # GeminiLLM wrapper
│   ├── state.py                             # QAState (TypedDict)
│   ├── nodes.py                             # QANodes — answer_node
│   ├── graph.py                             # QAGraph — builds & compiles StateGraph
│   ├── qa_runner.py                         # QARunner — console entry point
│   ├── app.py                               # QAApp — Gradio web UI
│   └── figure/                              # Auto-generated graph diagrams
├── basics-2-state-annotated-reducers/      # Series 1, Part 2
│   ├── config.py                            # Loads .env, defines Config
│   ├── llm.py                               # GeminiLLM wrapper
│   ├── state.py                             # TopicState with Annotated points field
│   ├── nodes.py                             # TopicNodes — expand_node, refine_node
│   ├── prompts/                             # LLM prompt templates
│   │   ├── expand_node.txt                  # Prompt for expand_node
│   │   └── refine_node.txt                  # Prompt for refine_node
│   ├── graph.py                             # TopicGraph — START → expand → refine → END
│   ├── topic_runner.py                      # TopicRunner — console entry point
│   └── app.py                               # TopicApp — Gradio web UI
├── requirements.txt                         # Shared dependencies
└── .env                                     # API keys (never commit)
```

---

## Running a Project

Each project has two entry points.

**Console runner** (the runner file is named after the project — e.g. `qa_runner.py`, `topic_runner.py`):

```bash
# Part 1
cd basics-1-stategraph-nodes-edges
python qa_runner.py

# Part 2
cd basics-2-state-annotated-reducers
python topic_runner.py
```

**Gradio web UI:**

```bash
# Part 1
cd basics-1-stategraph-nodes-edges
python app.py

# Part 2
cd basics-2-state-annotated-reducers
python app.py
```

The web UI starts a local server at `http://127.0.0.1:7860` and opens a chat interface in the browser.

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `langgraph` | 1.1.10 | Graph-based AI workflow framework |
| `langchain` | 1.2.17 | LLM abstraction layer |
| `langchain-google-genai` | 4.2.2 | Google Gemini integration |
| `python-dotenv` | 1.2.2 | Environment variable loading |
| `gradio` | 6.14.0 | Web UI for interactive demos |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Md Shafiqul Islam** — AI Engineer / LLM Specialist  
Blog: [shafiqulai.github.io](https://shafiqulai.github.io)
