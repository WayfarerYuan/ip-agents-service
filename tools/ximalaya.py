from langchain_core.tools import tool
from agent_service.skills.ximalaya import XimalayaSkill

# Default configuration for Huafei
DEFAULT_CONFIG = {
    "anchor_id": 41855169 # Huafei's ID
}

@tool
async def search_ximalaya(query_type: str = "tracks", page: int = 1) -> str:
    """
    Search for Ximalaya audio tracks or albums for the anchor.
    Args:
        query_type: "tracks" or "albums" (default: "tracks")
        page: Page number (default: 1)
    """
    skill = XimalayaSkill(config=DEFAULT_CONFIG)
    
    # Result is a dict containing metadata and a list of items
    result = await skill.execute({"type": query_type, "page": page})
    
    if isinstance(result, dict) and "error" in result:
        return f"Error fetching Ximalaya data: {result['error']}"
        
    if not result:
        return "No results found."
        
    output = f"### Ximalaya {query_type.capitalize()} (Page {page}):\n"
    
    items = []
    
    # Handle Tracks Structure
    if query_type == "tracks":
        # Key: trackDetailInfos
        items = result.get("trackDetailInfos", [])
        
        for i, item in enumerate(items):
            info = item.get("trackInfo", {})
            stat = item.get("statCountInfo", {})
            uri = item.get("pageUriInfo", {})
            
            title = info.get("title", "Unknown Title")
            play_count = stat.get("playCount", 0)
            url = uri.get("url", "")
            
            # Construct full URL if relative
            if url and not url.startswith("http"):
                url = f"https://m.ximalaya.com{url}"
                
            output += f"{i+1}. {title} (Plays: {play_count})\n   URL: {url}\n"

    # Handle Albums Structure
    else:
        # Key: albumBriefDetailInfos
        items = result.get("albumBriefDetailInfos", [])
        
        for i, item in enumerate(items):
            info = item.get("albumInfo", {})
            stat = item.get("statCountInfo", {})
            uri = item.get("pageUriInfo", {})
            
            title = info.get("title", "Unknown Title")
            play_count = stat.get("playCount", 0)
            url = uri.get("url", "")
            
            if url and not url.startswith("http"):
                url = f"https://m.ximalaya.com{url}"
                
            output += f"{i+1}. {title} (Plays: {play_count})\n   URL: {url}\n"
            
    if not items:
        return "No items found in the response."

    return output
