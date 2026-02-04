from langchain_core.tools import tool
import json

@tool
def display_card(card_type: str, data: dict) -> str:
    """
    Display a UI card to the user. Use this tool when you want to show structured information like a question, an assessment result, or a subscription prompt.
    
    Args:
        card_type: The type of card to display. Supported values: 
                   - 'question': For displaying a question with options.
                   - 'assessment': For displaying assessment feedback and next question preview.
                   - 'result': For displaying the final result of an assessment.
                   - 'subscription': For displaying a subscription prompt.
                   - 'briefing': For displaying a news briefing.
        data: The data required for the card.
    
    Returns:
        A JSON string representing the card data, formatted for the frontend.
    """
    # Construct the internal structure that the frontend expects inside the 'iting' event
    card_value = {
        "cardType": card_type,
        "cardData": data
    }
    
    # We return this structure. The server.py will wrap this in the full 'iting' protocol envelope.
    # Or we can return the full envelope here. 
    # Let's return the inner value for flexibility, and let server.py handle the protocol wrapper if possible.
    # But to make it distinct from other text, maybe we return a specific dict structure.
    
    return json.dumps(card_value, ensure_ascii=False)
