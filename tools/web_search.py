import requests
import json
from langchain_core.tools import tool
from agent_service.config import VOLC_SEARCH_API_KEY

@tool
def web_search(query: str) -> str:
    """
    Search the web for real-time information.
    Use this tool when you need to answer questions about current events, weather, stock prices, or any information not in your knowledge base.
    """
    url = "https://open.feedcoopapi.com/search_api/web_search"
    
    headers = {
        "Authorization": f"Bearer {VOLC_SEARCH_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Use "web" search type to get search results (title, summary, link)
    # Filter.NeedContent=True might provide full text if available/subscribed, 
    # but for basic RAG, snippets (summary) are often enough.
    payload = {
        "Query": query,
        "SearchType": "web",
        "Count": 8,
        # "Industry": "finance", # Removing industry restriction to broaden results
        "Filter": {
            "NeedContent": False,
            "NeedUrl": True
        }
    }
    
    try:
        print(f"[DEBUG] WebSearchTool calling: {url} with query: {query}")
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"[DEBUG] Search API Error: {response.status_code} - {response.text}")
            return f"Error: Search API returned {response.status_code}. Please check configuration."
            
        data = response.json()
        # print(f"[DEBUG] Search API Response: {str(data)[:200]}...")
        
        if data.get("code") != 0 and data.get("code") != "0": # Assuming 0 is success
             # Sometimes APIs return 200 OK but application level error
             if "msg" in data and data["msg"] != "success":
                 print(f"[DEBUG] Search API Logical Error: {data}")
        
        # Parse result
        # Structure based on debug script: 
        # Result -> WebResults -> list of {Title, Url, Snippet, Summary, PublishTime, SiteName}
        
        result_root = data.get("Result", {})
        if not result_root:
             # Try other common keys if 'Result' is missing
             result_root = data.get("data", {}).get("Result", {})
             
        web_results = result_root.get("WebResults", [])
             
        if not web_results:
            return "No results found."
            
        summary = ""
        for i, item in enumerate(web_results):
            title = item.get('Title', 'No Title')
            # Use Snippet as Summary if Summary is empty
            desc = item.get('Summary', '')
            if not desc:
                desc = item.get('Snippet', '')
            
            link = item.get('Url', '')
            site = item.get('SiteName', 'Unknown Source')
            time = item.get('PublishTime', '')
            
            summary += f"{i+1}. {title} ({site})\n   Time: {time}\n   Summary: {desc}\n   URL: {link}\n\n"
            
        return summary
        
    except Exception as e:
        print(f"[DEBUG] Search Exception: {e}")
        return f"Error performing search: {str(e)}"
