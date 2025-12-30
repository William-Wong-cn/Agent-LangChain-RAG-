import os
from tavily import TavilyClient

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query: str) -> str:
    response = tavily.search(query=query, max_results=3)
    return "\n".join([item["content"] for item in response["results"]])

