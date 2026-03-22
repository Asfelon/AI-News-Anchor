import chromadb
from datetime import datetime, timedelta
import urllib.parse
import requests
import xml.etree.ElementTree as ET
import uuid
from pathlib import Path

# ---------------------------------------------------------
# Database Initialization (Shared by ETL and TTL)
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = str(BASE_DIR / "news_vector_db")

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="news_archive")
topic_registry = chroma_client.get_or_create_collection(name="topic_registry")

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
RETENTION_DAYS = 90 
SCHEDULE_HOURS = 4

# ---------------------------------------------------------
# 1. TTL PIPELINE (Garbage Collection)
# ---------------------------------------------------------
def run_garbage_collection():
    print(f"\n[TTL MANAGER] -> Sweeping database for records older than {RETENTION_DAYS} days...")
    
    # Fetch all records
    results = collection.get(include=["metadatas"])
    
    # Extract the lists explicitly to satisfy Pylance type checking
    db_ids = results.get('ids')
    db_metadatas = results.get('metadatas')
    
    # Prove to Pylance that neither of these are None
    if not db_ids or not db_metadatas:
        print("[TTL MANAGER] -> Database is empty or missing metadata. Nothing to clean.")
        return

    ids_to_delete = []
    cutoff_date = datetime.now() - timedelta(days=RETENTION_DAYS)

    # Loop safely through the validated lists
    for doc_id, metadata in zip(db_ids, db_metadatas):
        ingestion_str = metadata.get("ingestion_date")
        
        if ingestion_str:
            ingestion_date = datetime.fromisoformat(str(ingestion_str))
            if ingestion_date < cutoff_date:
                ids_to_delete.append(doc_id)

    # Execute the deletion
    if ids_to_delete:
        print(f"[TTL MANAGER] -> Found {len(ids_to_delete)} expired records. Executing deletion...")
        collection.delete(ids=ids_to_delete)
        print(f"[TTL MANAGER] -> Success! Removed {len(ids_to_delete)} vectors.")
    else:
        print("[TTL MANAGER] -> No expired records found. Database is healthy.")

# ---------------------------------------------------------
# 2. ETL PIPELINE (Knowledge Base Refresh)
# ---------------------------------------------------------
def refresh_knowledge_base():
    """Automated background task to refresh all registered topics."""
    print("\n[ETL REFRESH] -> Starting knowledge base update...")
    try:
        registry_data = topic_registry.get()
        topics = registry_data.get("documents", [])
        
        if not topics:
            print("[ETL REFRESH] -> Topic registry is empty. Start chatting to build the queue!")
            return
            
        for topic in topics:
            print(f"[ETL REFRESH] -> Fetching latest news for: '{topic}'...")
            encoded_query = urllib.parse.quote(topic)
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
            
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            root = ET.fromstring(response.text)
            
            docs_to_save = []
            meta_to_save = []
            ids_to_save = []
            
            # Pull top 30 to ensure a deep historical archive
            for item in root.findall('.//item')[:30]:
                title = item.find('title').text if item.find('title') is not None else "No Title" # type: ignore
                link = item.find('link').text if item.find('link') is not None else "No Link" # type:ignore
                pub_date = item.find('pubDate').text if item.find('pubDate') is not None else "Unknown Date" # type:ignore
                
                doc_text = f"Headline: {title}\nPublished: {pub_date}\nLink: {link}\n"
                
                docs_to_save.append(doc_text)
                meta_to_save.append({
                    "source": link,
                    "ingestion_date": datetime.now().isoformat(),
                    "keyword_category": topic
                })
                ids_to_save.append(str(uuid.uuid4()))
                
            if docs_to_save:
                collection.add(documents=docs_to_save, metadatas=meta_to_save, ids=ids_to_save)
                
        print("[ETL REFRESH] -> Update complete. Database is fully synced.")
    except Exception as e:
        print(f"[ETL REFRESH ERROR] -> Pipeline failed: {e}")