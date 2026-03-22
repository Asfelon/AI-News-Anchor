import streamlit as st
from news_agent import agent_workflow # Assuming you import your compiled graph from nodes.py or agent.py

st.set_page_config(page_title="AI News Anchor", page_icon="📰", layout="centered")

st.title("📰 AI News Anchor")
st.caption("Powered by LangGraph, ChromaDB, and Llama 3")

# ---------------------------------------------------------
# 1. Session State Management
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your personal AI news anchor. What would you like to know about today?"}]

if "seen_articles" not in st.session_state:
    st.session_state.seen_articles = []

# ---------------------------------------------------------
# 2. Render the Chat History (This restores the full view!)
# ---------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------
# 3. Chat Input & Agent Execution
# ---------------------------------------------------------
if prompt := st.chat_input("Ask for news..."):
    
    # Restores the User Icon for the new message
    with st.chat_message("user"):
        st.markdown(prompt)
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Restores the AI Icon for the response
    with st.chat_message("assistant"):
        with st.spinner("Gathering live intel..."):
            
            # Feed BOTH the messages and the seen_articles into the agent
            initial_state = {
                "messages": st.session_state.messages,
                "seen_articles": st.session_state.seen_articles
            }
            final_response = ""
            
            try:
                # Stream the LangGraph workflow
                for event in agent_workflow.stream(initial_state): # type:ignore
                    for node_name, node_state in event.items():
                        
                        # MEMORY CAPTURE
                        if "seen_articles" in node_state:
                            new_article_count = len(node_state["seen_articles"])
                            print(f"[MEMORY CACHE] -> Saving {new_article_count} new articles to Session State!")
                            st.session_state.seen_articles.extend(node_state["seen_articles"])
                        
                        # Grab the final text from the output nodes
                        if node_name in ["summarizer", "direct_chat"]:
                            final_response = node_state["messages"][-1]["content"]
                            
                # Display and save the final response
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")