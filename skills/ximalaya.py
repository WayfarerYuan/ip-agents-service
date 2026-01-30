import aiohttp
from typing import Any, Dict, List
from agent_service.skills.base import BaseSkill

class XimalayaSkill(BaseSkill):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.anchor_id = self.config.get("anchor_id", 41855169) # Default to Huafei if not set

    @property
    def name(self) -> str:
        return "ximalaya_skill"

    @property
    def description(self) -> str:
        return "Fetch albums and tracks from Ximalaya for a specific anchor."

    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Input: {"type": "albums" | "tracks", "page": 1}
        """
        query_type = input_data.get("type", "tracks")
        page = input_data.get("page", 1)
        
        # tabType=0 for Albums, tabType=1 for Tracks
        tab_type = 0 if query_type == "albums" else 1
        
        url = f"https://m.ximalaya.com/m-revision/page/anchor/queryAnchorPage/{self.anchor_id}"
        params = {
            "pageSize": 20,
            "tabType": tab_type,
            "page": page
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Process data structure based on Ximalaya response
                    if data.get("ret") == 0:
                        # Extract relevant list
                        records = data.get("data", {}).get("anchorTrackPageRecords", {}) if tab_type == 1 else data.get("data", {}).get("anchorAlbumPageRecords", {})
                        # This structure needs verification based on actual API response from user provided URL
                        # User URL: https://m.ximalaya.com/m-revision/page/anchor/queryAnchorPage/41855169?pageSize=20&tabType=1
                        return records
                return {"error": f"Failed to fetch data: {resp.status}"}
