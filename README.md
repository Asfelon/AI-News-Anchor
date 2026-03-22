# 📰 AI News Anchor - Stateful RAG & Multi-Agent Architecture

An autonomous, multi-agent AI news system built with Python, LangGraph, ChromaDB, and Streamlit. This project goes beyond basic LLM wrappers by implementing an enterprise-grade, decoupled architecture featuring self-healing background data pipelines, hybrid Retrieval-Augmented Generation (RAG), and semantic cross-conversation memory.

## 🚀 Key Features

* **Multi-Agent Routing:** Utilizes LangGraph to dynamically route user inputs between a conversational agent and a highly analytical research agent based on intent.
* **Hybrid RAG Pipeline:** Intelligently queries a local ChromaDB vector database for historical context before seamlessly falling back to live web scraping (Google News RSS) when the cache is exhausted.
* **Semantic Deduplication & Memory:** Implements native Python text-matching (`difflib`) to compare the meaning of headlines. It actively prevents duplicate articles from being ingested and ensures users never see the same story twice in a single session, even if the source URLs differ.
* **Decoupled Background Jobs (ETL/TTL):** Runs an autonomous `scheduler.py` job completely separate from the UI. It features a "Harvester" to continuously refresh the database with new articles based on user interests, and a "Janitor" to enforce a Time-to-Live (TTL) retention policy, sweeping out expired vector embeddings.
* **Domain-Driven Design:** Structured as a modular Python package (`news_agent`) with strict separation of concerns between Models, Agents, Tools, and Execution scripts.

## 🧠 System Architecture

The project is structured into a modular library for maximum scalability:

```text
AI-News-Anchor/
├── .env                        # API Keys (Git-ignored)
├── requirements.txt            # Package dependencies
├── app.py                      # Streamlit UI Execution
├── scheduler.py                # Background Pipeline Execution
│
└── news_agent/                 # Main Library Package
    ├── __init__.py             # Exposes the library gateway
    │
    ├── agent/                  # Domain: The AI Brain
    │   ├── agent.py            # LangGraph StateGraph definition
    │   └── nodes.py            # LLM node logic and routing
    │
    ├── models/                 # Domain: Data Structures
    │   └── models.py           # TypedDicts and Pydantic schemas
    │
    └── tools/                  # Domain: Data Pipelines & Hands
        ├── tools.py            # Hybrid RAG and Semantic Memory logic
        ├── pipeline_tools.py   # ETL/TTL background functions
        └── tools_schema.yaml   # LLM function-calling schema