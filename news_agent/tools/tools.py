import urllib.parse
import requests
from xml.etree import ElementTree as ET
import chromadb
import uuid
from datetime import datetime
import threading 
import difflib 
from pathlib import Path

# ---------------------------------------------------------
# Database Initialization
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = str(BASE_DIR / "news_vector_db")

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="news_archive")
topic_registry = chroma_client.get_or_create_collection(name="topic_registry") 

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def register_unique_topic(keywords: str):
    """Checks if the topic already exists. If it's semantically new, adds it."""
    try:
        results = topic_registry.query(query_texts=[keywords], n_results=1)
        
        # FIX: Check if the inner list is actually populated before checking the distance
        if not results['distances'] or len(results['distances'][0]) == 0 or results['distances'][0][0] > 0.5:
            print(f"[TOPIC REGISTRY] -> New unique topic detected. Adding '{keywords}' to queue.")
            topic_registry.add(documents=[keywords], ids=[str(uuid.uuid4())])
        else:
            matched_topic = results['documents'][0][0] # type:ignore
            print(f"[TOPIC REGISTRY] -> Topic '{keywords}' is a duplicate of '{matched_topic}'. Skipping.")
            
    except Exception as e:
        print(f"[TOPIC REGISTRY ERROR] -> {e}")

def background_db_insert(documents, metadatas, ids):
    """Runs invisibly to save bulk data."""
    try:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"[BACKGROUND THREAD] -> Success! Bulk dump saved to ChromaDB.")
    except Exception as e:
        print(f"\n[BACKGROUND THREAD ERROR] -> {e}")

def extract_headline(doc_text: str) -> str:
    """Pulls just the headline out of the formatted text to prevent URLs from ruining the similarity score."""
    for line in doc_text.split('\n'):
        if line.startswith("Headline: "):
            return line.replace("Headline: ", "").strip()
    return doc_text # Fallback if formatting is weird

def is_already_seen(new_article: str, seen_articles: list[str]) -> bool:
    """Checks if the HEADLINE is semantically similar to ANY headline the user has already read."""
    new_headline = extract_headline(new_article)
    
    for seen_text in seen_articles:
        seen_headline = extract_headline(seen_text)
        similarity = difflib.SequenceMatcher(None, new_headline, seen_headline).ratio()
        if similarity > 0.55:
            return True
    return False

def get_distinct_and_verbose(raw_articles: list[str], top_n: int = 3) -> list[str]:
    """Clusters similar articles based on HEADLINES and keeps the longest (most verbose) version."""
    distinct_stories = []
    
    for article in raw_articles:
        is_duplicate = False
        article_headline = extract_headline(article)
        
        for i, existing_story in enumerate(distinct_stories):
            existing_headline = extract_headline(existing_story)
            similarity = difflib.SequenceMatcher(None, article_headline, existing_headline).ratio()
            
            if similarity > 0.55:
                is_duplicate = True
                if len(article) > len(existing_story):
                    distinct_stories[i] = article
                break
                
        if not is_duplicate:
            distinct_stories.append(article)
            
    return distinct_stories[:top_n]

# ---------------------------------------------------------
# Tool Definition: Hybrid RAG Search
# ---------------------------------------------------------
def hybrid_news_search(keywords: str, seen_articles: list[str] | None = None) -> dict:
    """Uses Semantic Memory to prevent repeating news concepts in the same chat session."""
    if seen_articles is None: seen_articles = []
    register_unique_topic(keywords)
    print(f"\n[HYBRID SEARCH] -> Querying DB. (User has already read {len(seen_articles)} articles)")
    
    # --- STEP 1: Search Internal DB ---
    try:
        results = collection.query(query_texts=[keywords], n_results=15) 
        
        if results['documents'] and results['distances'] and len(results['distances'][0]) > 0:
            if results['distances'][0][0] < 1.2:
                unseen_db_docs = [doc for doc in results['documents'][0] if not is_already_seen(doc, seen_articles)]
                best_articles = get_distinct_and_verbose(unseen_db_docs, top_n=3)
                
                if best_articles:
                    print(f"[HYBRID SEARCH] -> Cache Hit! Found UNSEEN historical data.")
                    return {"text": "\n".join(best_articles), "new_articles": best_articles}
                else:
                    print(f"[HYBRID SEARCH] -> Cache Exhausted. User has read all DB concepts on this topic.")
    except Exception as e:
         print(f"[HYBRID SEARCH ERROR] -> {e}")
            
    # --- STEP 2: Cache Miss / Exhausted - Fetch Bulk Live Data ---
    print(f"[HYBRID SEARCH] -> Fetching fresh live data from the web...")
    search_text = "No results found."
    new_articles_for_memory = []
    
    try:
        encoded_query = urllib.parse.quote(keywords)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        root = ET.fromstring(response.text)
        all_items = root.findall('.//item')
        
        if not all_items:
            return {"text": "The RSS feed returned zero results.", "new_articles": []}

        unseen_raw_results = []
        docs_to_save = []
        meta_to_save = []
        ids_to_save = []
        
        for item in all_items[:40]:
            title = item.find('title').text if item.find('title') is not None else "No Title" # type:ignore
            link = item.find('link').text if item.find('link') is not None else "No Link" # type:ignore
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else "Unknown Date" # type:ignore
            
            doc_text = f"Headline: {title}\nPublished: {pub_date}\nLink: {link}\n"
            
            # Save ALL to the database
            docs_to_save.append(doc_text)
            meta_to_save.append({
                "source": link,
                "ingestion_date": datetime.now().isoformat(),
                "keyword_category": keywords
            })
            ids_to_save.append(str(uuid.uuid4()))
            
            # Only consider it for the UI if it passes Semantic Memory check
            if not is_already_seen(doc_text, seen_articles):
                unseen_raw_results.append(doc_text)
            
        if docs_to_save:
            threading.Thread(target=background_db_insert, args=(docs_to_save, meta_to_save, ids_to_save)).start()
            
        best_web_articles = get_distinct_and_verbose(unseen_raw_results, top_n=3)
        
        if best_web_articles:
            search_text = "\n".join(best_web_articles)
            new_articles_for_memory = best_web_articles
        else:
            search_text = "No new articles found on this topic."
            
    except Exception as e:
        print(f"\n[AGENT TOOL ERROR] {e}\n")
        search_text = f"System Error: {e}"
        
    return {"text": search_text, "new_articles": new_articles_for_memory}