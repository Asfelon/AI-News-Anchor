from typing import TypedDict, Annotated, Literal, NotRequired
import operator
from pydantic import BaseModel, Field

class State(TypedDict):
    messages: Annotated[list[dict], operator.add]
    seen_articles: Annotated[list[str], operator.add] # <-- CRITICAL FOR MEMORY
    intermediate_reasoning: NotRequired[str]  
    extracted_keywords: NotRequired[str]
    search_results: NotRequired[str]

class RouteDecision(BaseModel):
    decision: Literal["researcher", "direct_chat"] = Field(
        description=(
            "Route to 'researcher' ONLY if the user explicitly asks for information, news, updates, or facts. "
            "Route to 'direct_chat' for greetings, casual conversation, OR if the user is just sharing an opinion or expressing an emotion about a topic without asking for factual updates."
        )
    )