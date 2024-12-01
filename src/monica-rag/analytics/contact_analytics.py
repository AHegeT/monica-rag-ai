from typing import Dict, List, Optional, Union
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class ContactAnalytics:
    def __init__(self, contacts: List[Dict], contact_fields: Dict[int, List[Dict]]):
        """
        Initialize ContactAnalytics with contacts and their fields.
        
        Args:
            contacts: List of contact dictionaries from Monica API
            contact_fields: Dictionary mapping contact IDs to their contact fields
        """
        self.contacts = contacts
        self.contact_fields = contact_fields
        self._total_contacts = len(contacts)

    def get_field_completion_rate(self, field_name: str) -> Dict[str, Union[float, int]]:
        """
        Calculate completion rate for a specific contact field.
        
        Args:
            field_name: Name of the contact field to check
            
        Returns:
            Dict containing completion rate and counts
        """
        completed = 0
        missing = 0
        
        for contact in self.contacts:
            contact_id = contact['id']
            fields = self.contact_fields.get(contact_id, [])
            
            # Check if the field exists and has a value
            field_exists = any(
                field.get('content_type', {}).get('name', '').lower() == field_name.lower() 
                and field.get('data', {}).get('value')
                for field in fields
            )
            
            if field_exists:
                completed += 1
            else:
                missing += 1
        
        completion_rate = (completed / self._total_contacts * 100) if self._total_contacts > 0 else 0
        
        return {
            'completion_rate': round(completion_rate, 2),
            'completed': completed,
            'missing': missing,
            'total': self._total_contacts
        }

    def get_field_value_distribution(self, field_name: str) -> Dict[str, int]:
        """
        Get distribution of values for a specific contact field.
        
        Args:
            field_name: Name of the contact field to analyze
            
        Returns:
            Dict mapping values to their frequency
        """
        values = []
        for contact in self.contacts:
            contact_id = contact['id']
            fields = self.contact_fields.get(contact_id, [])
            
            # Get all values for this field
            field_values = [
                str(field.get('data', {}).get('value'))
                for field in fields
                if field.get('content_type', {}).get('name', '').lower() == field_name.lower()
                and field.get('data', {}).get('value')
            ]
            
            values.extend(field_values)
        
        return dict(Counter(values))

    def get_multiple_fields_completion(self, fields: List[str]) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Calculate completion rates for multiple fields.
        
        Args:
            fields: List of field paths to check
            
        Returns:
            Dict mapping field names to their completion statistics
        """
        return {
            field: self.get_field_completion_rate(field)
            for field in fields
        }

    def _get_nested_field(self, data: Dict, field_path: str) -> Optional[any]:
        """
        Get value from nested dictionary using dot notation.
        
        Args:
            data: Dictionary to search in
            field_path: Dot-separated path to the field
            
        Returns:
            Field value if found, None otherwise
        """
        current = data
        for part in field_path.split('.'):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _is_field_completed(self, value: any) -> bool:
        """
        Check if a field value should be considered completed.
        
        Args:
            value: Field value to check
            
        Returns:
            Boolean indicating if field is completed
        """
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        if isinstance(value, (list, dict)) and not value:
            return False
        return True

    def get_field_value_distribution(self, field_path: str) -> Dict[str, int]:
        """
        Get distribution of values for a specific field.
        
        Args:
            field_path: Dot-separated path to the field
            
        Returns:
            Dict mapping values to their frequency
        """
        values = []
        for contact in self.contacts:
            value = self._get_nested_field(contact, field_path)
            if self._is_field_completed(value):
                values.append(str(value))
        
        return dict(Counter(values))