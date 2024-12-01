from typing import Dict, List

def create_contact_text(contact: Dict) -> str:
    """
    Create a text representation of contact information for embedding.
    
    Args:
        contact (Dict): Contact information dictionary from Monica API
        
    Returns:
        str: Formatted text representation of the contact
    """
    parts = []
    
    # Basic information
    name = f"{contact.get('first_name', '')} {contact.get('last_name', '')}".strip()
    if name:
        parts.append(f"Name: {name}")
    
    # Work information
    if contact.get('job'):
        parts.append(f"Job: {contact['job']}")
    if contact.get('company'):
        parts.append(f"Company: {contact['company']}")
    
    # Contact information (you can expand this based on your Monica API response)
    if contact.get('email'):
        parts.append(f"Email: {contact['email']}")
    if contact.get('phone'):
        parts.append(f"Phone: {contact['phone']}")
        
    # Notes (if available in your API response)
    if contact.get('notes'):
        parts.append(f"Notes: {contact['notes']}")
        
    return "\n".join(filter(None, parts))

def format_search_results(results: List[Dict]) -> str:
    """
    Format search results for display.
    
    Args:
        results (List[Dict]): List of search results with contact info and similarity scores
        
    Returns:
        str: Formatted string of results
    """
    output = []
    for i, result in enumerate(results, 1):
        contact = result['contact']
        similarity = result['similarity']
        
        name = f"{contact.get('first_name', '')} {contact.get('last_name', '')}".strip()
        output.append(f"\n{i}. {name}")
        output.append(f"   Similarity: {similarity:.3f}")
        
        if contact.get('job'):
            output.append(f"   Job: {contact['job']}")
        if contact.get('company'):
            output.append(f"   Company: {contact['company']}")
            
    return "\n".join(output)