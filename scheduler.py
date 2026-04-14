import schedule
import time

# Import your pipeline functions and the dynamic variable
from news_agent.tools.pipeline_tools import run_garbage_collection, refresh_knowledge_base, SCHEDULE_HOURS

def execute_pipeline():
    """The master pipeline execution order."""
    print("\n" + "="*50)
    print("🔄 STARTING AUTOMATED PIPELINE CYCLE")
    print("="*50)
    
    run_garbage_collection()
    refresh_knowledge_base()
    
    print("\n" + "="*50)
    print(f"⏸️ PIPELINE CYCLE COMPLETE. Next run scheduled in {SCHEDULE_HOURS} hours.")
    print("="*50 + "\n")

# ---------------------------------------------------------
# The Schedule Configuration
# ---------------------------------------------------------
if __name__ == "__main__":
    # Convert the hours (float or int) into exact integer minutes
    schedule_minutes = int(SCHEDULE_HOURS * 60)
    
    print("⏰ Starting Master Data Scheduler...")
    print(f"⚙️ Configuration: Running pipelines every {SCHEDULE_HOURS} hours ({schedule_minutes} minutes).")
    
    # Use .minutes instead of .hours to completely avoid the float issue!
    schedule.every(schedule_minutes).minutes.do(execute_pipeline)
    
    # Run it once immediately upon startup
    execute_pipeline()
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n🛑 Scheduler terminated by user.")