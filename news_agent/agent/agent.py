# --------------------------------------------------------------------------------
# Agent AI Imports (Groq API)
# --------------------------------------------------------------------------------
from .nodes import route_user_input, researcher_node, execute_search_node, summarizer_node, direct_chat_node
from ..models.models import State
from langgraph.graph import StateGraph, START, END


workflow = StateGraph(State)

workflow.add_node("researcher", researcher_node)
workflow.add_node("execute_search", execute_search_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("direct_chat", direct_chat_node)

# The Conditional Fork
workflow.add_conditional_edges(
    START,
    route_user_input, 
    {
        "researcher": "researcher",   
        "direct_chat": "direct_chat"  
    }
)

# Standard Edges
workflow.add_edge("researcher", "execute_search")
workflow.add_edge("execute_search", "summarizer")
workflow.add_edge("summarizer", END)
workflow.add_edge("direct_chat", END)

app = workflow.compile()

# --- Execution ---
if __name__ == "__main__":
    print("Testing Pipeline...\n")
    
    # Test 1: Casual Chat (Should route to DIRECT_CHAT)
    chat_input: State = {
        "messages": [{"role": "user", "content": "Hi there! Who are you?"}]
    } # type:ignore
    
    print("--- Running Test 1: Casual Chat ---")
    for event in app.stream(chat_input):
        for node_name, node_state in event.items():
            if node_name == "direct_chat":
                print("\nAgent Response:\n", node_state["messages"][-1]["content"])
                print("\n" + "="*50 + "\n")

    # Test 2: Factual Query (Should route to RESEARCHER)
    news_input: State = {
        "messages": [{"role": "user", "content": "What's the latest news regarding infrastructure projects in Thane?"}]
    } # type:ignore
    
    print("--- Running Test 2: RAG Pipeline ---")
    for event in app.stream(news_input):
        for node_name, node_state in event.items():
            if node_name == "summarizer":
                print("\nFinal Broadcast:\n", node_state["messages"][-1]["content"])