from langchain_core.tools import tool
import json

@tool
def display_card(card_type: str, data: dict) -> str:
    """
    Display a UI card to the user. Use this tool to present structured interactions.

    Supported card_type and data schema:

    1. 'interactive_options': For quizzes, questions, or branch selections.
       Data Schema:
       {
           "title": "Question text",
           "options": [
               {"label": "A", "text": "Option A text", "value": "value_a"},
               {"label": "B", "text": "Option B text", "value": "value_b"}
           ],
           "multiple": false
       }

    2. 'profile_card': For displaying character stats, role info, or user attributes.
       Data Schema:
       {
           "title": "Name/Role",
           "subtitle": "Description",
           "avatar": "http://...", 
           "tags": ["Tag1", "Tag2"],
           "attributes": [{"key": "Strength", "value": "100"}]
       }

    3. 'info_card': For knowledge display, analysis results, or general info.
       Data Schema:
       {
           "title": "Title",
           "content": "Markdown supported content",
           "image_url": "http://... (optional)",
           "link": {"text": "More", "url": "http://..."} (optional)
       }

    Args:
        card_type: The type of card (interactive_options, profile_card, info_card).
        data: The content dictionary matching the schema.

    Returns:
        JSON string to be sent to client.
    """
    # Construct standard internal structure
    # This structure will be wrapped in 'iting' protocol by server.py
    # Structure: { "cardType": "...", "cardData": { ... } }
    card_value = {
        "cardType": card_type,
        "cardData": data
    }
    
    return json.dumps(card_value, ensure_ascii=False)
