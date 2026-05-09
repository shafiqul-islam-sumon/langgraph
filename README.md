# LangGraph Blog Series

[![Python](https://img.shields.io/badge/Python-3.12-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1.10-10b981?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

Companion source code for the **LangGraph** blog series published at [shafiqulai.github.io](https://shafiqulai.github.io).

All implementations use **Python 3.12**, **LangGraph**, and **Google Gemini** as the LLM. Every project follows an OOP structure — no standalone scripts.

---

## Series 1 — LangGraph Basics

LangGraph applications — no matter how complex — are built from three primitives: **State**, **Nodes**, and **Edges**. Before writing a chatbot, a RAG pipeline, or a multi-agent system, those primitives need to be solid. This series covers each one in depth, starting from the simplest possible graph and adding one concept at a time until the full LangGraph toolkit is in place.

| # | Folder | Concept | Blog | README |
|---|--------|---------|------|--------|
| Part 1 | [`basics-1-stategraph-nodes-edges/`](basics-1-stategraph-nodes-edges/) | StateGraph, Nodes & Edges | [Read →](https://shafiqulai.github.io/blogs/blog_8.html) | [README →](basics-1-stategraph-nodes-edges/README.md) |
| Part 2 | [`basics-2-state-annotated-reducers/`](basics-2-state-annotated-reducers/) | State, Annotated Fields & Custom Reducers | [Read →](https://shafiqulai.github.io/blogs/blog_9.html) | [README →](basics-2-state-annotated-reducers/README.md) |
| Part 3 | [`basics-3-conditional-edges/`](basics-3-conditional-edges/) | Conditional Edges & Routing Logic | [Read →](https://shafiqulai.github.io/blogs/blog_10.html) | [README →](basics-3-conditional-edges/README.md) |
| Part 4 | [`basics-4-checkpointers-memory-streaming/`](basics-4-checkpointers-memory-streaming/) | Checkpointers, Memory & Streaming | [Read →](https://shafiqulai.github.io/blogs/blog_11.html) | [README →](basics-4-checkpointers-memory-streaming/README.md) |
| Part 5 | `basics-5-tools-toolnode/` | Tools, ToolNode & Prebuilt Components | Coming soon | — |
| Part 6 | `basics-6-subgraphs-hitl/` | Subgraphs, Interrupt & Human-in-the-Loop | Coming soon | — |

### Part 1 — StateGraph, Nodes & Edges

Introduces the three primitives every LangGraph graph is built from. Builds a **Q&A bot** (single node, single edge) powered by Google Gemini — the smallest possible complete graph, with a console runner and a Gradio web UI.

→ [Full details in basics-1-stategraph-nodes-edges/README.md](basics-1-stategraph-nodes-edges/README.md)

### Part 2 — State, Annotated Fields & Custom Reducers

Covers LangGraph's default **last-write-wins** merge rule and how `Annotated` fields with reducer functions override it. Builds a **Topic Expander** where two nodes both write to the same `points` list — `operator.add` ensures both contributions accumulate rather than one overwriting the other.

→ [Full details in basics-2-state-annotated-reducers/README.md](basics-2-state-annotated-reducers/README.md)

### Part 3 — Conditional Edges & Routing Logic

Introduces **conditional edges** — the mechanism that lets a graph branch at runtime based on state. A router function reads the current state and returns the name of the next node. Builds a **Customer Support Router** that classifies incoming messages and routes them to dedicated billing, technical, or general support nodes.

→ [Full details in basics-3-conditional-edges/README.md](basics-3-conditional-edges/README.md)

### Part 4 — Checkpointers, Memory & Streaming

Introduces **persistent memory** using LangGraph checkpointers. A single argument at compile time — `graph.compile(checkpointer=...)` — turns any stateless graph into one that remembers every prior turn. Covers `MemorySaver` (in-RAM, development) and `SqliteSaver` (file-backed, production), `thread_id` isolation, `get_state()` / `get_state_history()`, and three streaming modes. Builds a **Personal Study Buddy** where Alice holds a multi-turn session about recursion, Bob starts an isolated session, and thread isolation is proven live.

→ [Full details in basics-4-checkpointers-memory-streaming/README.md](basics-4-checkpointers-memory-streaming/README.md)

---

## Series 2 — LangGraph: From Zero to Production

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

All projects share a single virtual environment and `requirements.txt`.

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

Create a `.env` file in this folder (`langgraph/.env`):

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
├── basics-1-stategraph-nodes-edges/    # Part 1 — Q&A bot
│   ├── README.md                        # Concepts, code walkthrough, how to run
│   ├── state.py                         # QAState (TypedDict)
│   ├── nodes.py                         # QANodes — answer_node
│   ├── graph.py                         # QAGraph — START → answer → END
│   ├── qa_runner.py                     # Console entry point
│   ├── app.py                           # Gradio web UI
│   ├── config.py / llm.py               # Shared config and LLM wrapper
│   └── figure/                          # Auto-generated Mermaid diagrams
├── basics-2-state-annotated-reducers/  # Part 2 — Topic Expander
│   ├── README.md                        # Concepts, code walkthrough, how to run
│   ├── state.py                         # TopicState with Annotated points field
│   ├── nodes.py                         # TopicNodes — expand_node, refine_node
│   ├── graph.py                         # TopicGraph — START → expand → refine → END
│   ├── topic_runner.py                  # Console entry point
│   ├── app.py                           # Gradio web UI
│   ├── prompts/                         # LLM prompt templates (one file per node)
│   ├── config.py / llm.py               # Shared config and LLM wrapper
│   └── figure/                          # Auto-generated Mermaid diagrams
├── basics-3-conditional-edges/         # Part 3 — Customer Support Router
│   ├── README.md                        # Concepts, code walkthrough, how to run
│   ├── state.py                         # SupportState — message, category, response
│   ├── router.py                        # route_by_category() — reads state, returns node name
│   ├── nodes.py                         # SupportNodes — classify + 3 support nodes
│   ├── graph.py                         # SupportGraph — conditional edges via add_conditional_edges
│   ├── support_runner.py                # Console entry point
│   ├── app.py                           # Gradio web UI
│   ├── prompts/                         # LLM prompt templates (one file per node)
│   ├── config.py / llm.py               # Shared config and LLM wrapper
│   └── figure/                          # Auto-generated Mermaid diagrams
├── basics-4-checkpointers-memory-streaming/  # Part 4 — Personal Study Buddy
│   ├── README.md                             # Concepts, code walkthrough, how to run
│   ├── state.py                              # StudyState — messages with add_messages reducer
│   ├── nodes.py                              # StudyBuddyNodes — study_buddy_node
│   ├── graph.py                              # StudyBuddyGraph — compiled with checkpointer
│   ├── study_runner.py                       # Console entry point with multi-turn demo
│   ├── app.py                                # Gradio Blocks with gr.State for thread isolation
│   ├── prompts/                              # LLM prompt templates (study_buddy.txt)
│   ├── config.py / llm.py                    # Shared config and LLM wrapper
│   └── figure/                               # Auto-generated Mermaid diagrams
├── requirements.txt                     # Shared dependencies
└── .env                                 # API keys (never commit)
```

---

## Running a Project

Each project has two entry points. Activate the virtual environment first, then `cd` into the project folder.

**Console runner:**

```bash
# Part 1
cd basics-1-stategraph-nodes-edges && python qa_runner.py

# Part 2
cd basics-2-state-annotated-reducers && python topic_runner.py

# Part 3
cd basics-3-conditional-edges && python support_runner.py

# Part 4
cd basics-4-checkpointers-memory-streaming && python study_runner.py
```

**Gradio web UI:**

```bash
# Any part
cd basics-N-<folder-name> && python app.py
```

The web UI starts at `http://127.0.0.1:7860`.

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
