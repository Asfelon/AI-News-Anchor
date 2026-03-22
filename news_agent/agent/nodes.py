# --------------------------------------------------------------------------------
# Environment Imports
# --------------------------------------------------------------------------------
import os
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------------------------------------
# Agent AI Imports (Groq API)
# --------------------------------------------------------------------------------
from groq import Groq
from ..tools.tools import hybrid_news_search

# --------------------------------------------------------------------------------
# Data Model Imports
# --------------------------------------------------------------------------------
import json
from ..models.models import State, RouteDecision
from typing import Literal, Any

# --------------------------------------------------------------------------------
# Other APIs
# --------------------------------------------------------------------------------
import yaml
from pathlib import Path

SEED = 42
CURRENT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = CURRENT_DIR.parent / "tools"
YAML_PATH = TOOLS_DIR / "tools_schema.yaml"
with open(YAML_PATH, "r") as yaml_file:
    TOOLS_SCHEMA = yaml.safe_load(yaml_file)

if TOOLS_SCHEMA is None:
    raise RuntimeError("Critical Configuration Missing: Tools Schema not loaded properly.")

api_key = os.getenv('GROQ_API_KEY')
llm_client = Groq(api_key=api_key)

# ---------------------------------------------------------
# Node 1: "The Router"
# ---------------------------------------------------------
def route_user_input(
        state: State,
        default_model: str = "llama-3.1-8b-instant",
        IS_DEVELOPMENT: bool = False
    ) -> Literal["researcher", "direct_chat"]:
    
    model_config: dict[str, Any] = {
        "model": default_model,
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }

    if IS_DEVELOPMENT:
        model_config["seed"] =  SEED

    user_message = state["messages"][-1]["content"]
    schema_str = json.dumps(RouteDecision.model_json_schema(), indent=2)
    
    system_prompt = f"""
    Analyze the user's input and decide the routing path.
    Respond strictly in JSON matching this schema:
    {schema_str}
    """
    
    response = llm_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        **model_config
    )
    
    try:
        decision_data = RouteDecision.model_validate_json(response.choices[0].message.content)  # type: ignore
        print(f"\n[ROUTER] -> Sending user to: {decision_data.decision.upper()}")
        return decision_data.decision
    except Exception as e:
        print(f"\n[ROUTER ERROR] Defaulting to researcher. Error: {e}")
        return "researcher"
    
# ---------------------------------------------------------
# Node 2: "The Researcher" (Think + Call Tool)
# ---------------------------------------------------------
def researcher_node(
        state: State,
        default_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        IS_DEVELOPMENT: bool = False
    ) -> dict:

    # 1. Setup Config (No JSON Mode needed for native tool calling!)
    model_config = {
        "model": default_model,
        "temperature": 0.0,
        "tools": [TOOLS_SCHEMA["news_tool_schema"]],
        "tool_choice": "auto" # Fixed typo
    }

    if IS_DEVELOPMENT:
        model_config["seed"] = SEED
    
    user_message = state["messages"][-1]["content"]
    
    # 2. Simplified Prompt
    system_prompt = "You are a research agent. Use the fetch_news_rss tool to search the web for the user's query. Extract 3 to 5 highly specific keywords to use as your search argument."

    response = llm_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        **model_config
    )
    
    message = response.choices[0].message
    
    # 3. Format the Groq message for LangGraph State
    assistant_msg = {
        "role": "assistant",
        "content": message.content or ""
    }
    
    # 4. Attach the tool calls if the LLM decided to search
    if message.tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            } for tc in message.tool_calls
        ]

    return {"messages": [assistant_msg]}

# ---------------------------------------------------------
# Node 2.5: "The Tool Executor" (Runs the Python function)
# ---------------------------------------------------------
def execute_search_node(state: State) -> dict:
    print("\n[TOOL EXECUTOR] -> Intercepting LLM request...")
    last_message = state["messages"][-1]
    
    # 1. Safety check: Did the LLM actually ask for a tool?
    tool_calls = last_message.get("tool_calls", [])
    if not tool_calls:
        print("[TOOL EXECUTOR] -> No tool requested. Skipping.")
        return {"search_results": "No search was performed."}
        
    # 2. Extract the instructions the LLM generated
    tool_call = tool_calls[0]
    arguments = json.loads(tool_call["function"]["arguments"])
    keywords = arguments.get("keywords", "")
    
    # 3. RUN THE ACTUAL PYTHON FUNCTION
    seen_articles = state.get("seen_articles", [])
    print(f"[TOOL EXECUTOR] -> Running hybrid_news_search('{keywords}')")
    tool_result = hybrid_news_search(keywords, seen_articles)

    search_text = tool_result["text"]
    new_articles = tool_result["new_articles"]    
    
    # 4. Format the response so the LLM knows it's the result of the tool
    tool_message = {
        "role": "tool",
        "content": search_text,
        "tool_call_id": tool_call["id"],
        "name": tool_call["function"]["name"]
    }
    
    return {
        "messages": [tool_message],
        "search_results": search_text,
        "extracted_keywords": keywords,
        "seen_articles": new_articles
    }

# ---------------------------------------------------------
# Node 3: "The Anchor" (Summarize Live Data)
# ---------------------------------------------------------
def summarizer_node(
        state: State,
        default_model: str = "llama-3.3-70b-versatile",
        IS_DEVELOPMENT: bool = False
    ) -> dict[str, list[dict[str, str | None]]]:

    model_config: dict[str, Any] = {
        "model": default_model
    }

    if IS_DEVELOPMENT:
        model_config["seed"] =  SEED
    
    live_data = state.get("search_results", "")
    
    system_prompt = f"""
    You are a professional news anchor and researcher. Read the following LIVE DATA pulled directly from the web.
    
    You must format your response EXACTLY according to the structural template below.
    
    ### 📰 The Briefing
    [Provide a bulleted list. Each bullet MUST start with a short, bolded **Headline**, followed by a concise 1-2 sentence summary of the core event.]
    
    ### 🗞️ Detailed Coverage
    [For every single bold **Headline** created in 'The Briefing', create a matching sub-header here. 
    Under each sub-header, write a detailed paragraph expanding on the context. 
    At the end of each detailed section, you MUST provide the exact URL link from the LIVE DATA using the format: "🔗 Source: [Read Article](URL)".]
    
    CRITICAL RULES:
    1. Do not make up, guess, or hallucinate URLs. If a link is not explicitly provided in the LIVE DATA, write "🔗 Source: Link unavailable".
    2. The bolded headlines in The Briefing MUST perfectly match the sub-headers in Detailed Coverage.
    3. If the LIVE DATA indicates an error or no results, completely ignore the template. Instead, apologize to the user and state that no information is available.
    
    LIVE DATA:
    {live_data}
    """

    # We switch the prompt from "Please summarize the latest updates" 
    # to simply passing the user's original message so the context is preserved.
    user_original_prompt = state["messages"][0]["content"]
    
    response = llm_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_original_prompt}
        ],
        **model_config
    )
    
    final_summary = response.choices[0].message.content or ""
    return {"messages": [{"role": "assistant", "content": final_summary}]}

# ---------------------------------------------------------
# Node 4: "The Conversationalist" (No Search)
# ---------------------------------------------------------
def direct_chat_node(
        state: State,
        IS_DEVELOPMENT: bool = False
    ) -> dict[str, list[dict[str, str | None]]]:
    user_message = state["messages"][-1]["content"]
    
    response = llm_client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[
            {"role": "system", "content": "You are a helpful AI news anchor. Respond conversationally and warmly."},
            {"role": "user", "content": user_message}
        ]
    )
    
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}

# --- Execution ---
if __name__ == "__main__":

    print('This file is not meant to be executed standalone. Please look for agent.py file for the Agent Execution.')