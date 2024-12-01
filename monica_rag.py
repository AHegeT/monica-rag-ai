from ssl import SSLError
import requests
from typing import Dict, List
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
from dotenv import load_dotenv
import os

load_dotenv()
base_url = os.getenv('MONICA_PROD_BASE_URL') if os.getenv('ENVIRONMENT') == 'prod' else os.getenv('MONICA_DEV_BASE_URL')
token = os.getenv('MONICA_PROD_API_TOKEN') if os.getenv('ENVIRONMENT') == 'prod' else os.getenv('MONICA_DEV_API_TOKEN')

class MonicaRAGStorage:
    def __init__(self, db_path: str = 'monica_rag.db'):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    contact_id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    contact_data TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def save_embedding(self, contact_id: int, embedding: np.ndarray, contact_data: Dict):
        """Save or update an embedding and contact data."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO embeddings (contact_id, embedding, contact_data, last_updated)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                contact_id,
                embedding.tobytes(),
                json.dumps(contact_data)
            ))
    
    def get_all_embeddings(self) -> Dict[int, tuple]:
        """Retrieve all embeddings and contact data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT contact_id, embedding, contact_data FROM embeddings")
            results = {}
            for contact_id, embedding_bytes, contact_data in cursor:
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32).copy()  # Specify dtype
                results[contact_id] = (embedding, json.loads(contact_data))
            return results

    def get_last_updated(self) -> str:
        """Get the timestamp of the most recent update."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT MAX(last_updated) FROM embeddings")
            return cursor.fetchone()[0]

class MonicaAPIClient:
    def __init__(self, base_url: str, token: str, verify_ssl: bool = True):
        """
        Initialize the Monica API client
        
        Args:
            base_url: Base URL of your Monica instance
            token: API token for authentication
            verify_ssl: Whether to verify SSL certificates. Set to False only in development
        """
        self.base_url = base_url.rstrip('/')
        if not self.base_url.startswith('https://'):
            self.base_url = f'https://{self.base_url}'
        
        self.token = token
        self.verify_ssl = verify_ssl
        
        if not verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    # def create_custom_field(self, contact_field_type_name, contact_id=None):
    #     """
    #     Create a custom field type and optionally associate it with a contact
        
    #     Args:
    #         contact_field_type_name (str): Name of the custom field
    #         contact_id (int, optional): ID of the contact to associate the field with
        
    #     Returns:
    #         dict: API response
    #     """
    #     # First create the custom field type
    #     endpoint = f"{self.base_url}/customfieldtypes"
    #     payload = {
    #         "name": contact_field_type_name,
    #         "protocol": "text"  # Can be: text, number, date, boolean
    #     }
        
    #     response = requests.post(endpoint, headers=self.headers, json=payload)
    #     if response.status_code != 201:
    #         raise Exception(f"Failed to create custom field type: {response.text}")
        
    #     field_type = response.json()['data']
        
    #     # If a contact_id is provided, create a custom field for that contact
    #     if contact_id:
    #         contact_endpoint = f"{self.base_url}/contacts/{contact_id}/contactfields"
    #         contact_payload = {
    #             "contact_field_type_id": field_type['id'],
    #             "data": ""  # Initial value can be set here
    #         }
            
    #         contact_response = requests.post(contact_endpoint, headers=self.headers, json=contact_payload)
    #         if contact_response.status_code != 201:
    #             raise Exception(f"Failed to associate field with contact: {contact_response.text}")
            
    #         return contact_response.json()
        
    #     return field_type

    def list_custom_fields(self):
        """Get custom field types."""
        return self._make_request('GET', 'customfieldtypes')['data']
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make an HTTP request with error handling"""
        try:
            response = self.session.request(
                method,
                f'{self.base_url}/api/{endpoint.lstrip("/")}',
                verify=self.verify_ssl,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except SSLError as e:
            raise ConnectionError(
                "SSL verification failed. If you're using a self-signed certificate, "
                "you may need to set verify_ssl=False (only do this in development)."
            ) from e
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"API request failed: {str(e)}") from e

    def get_contacts(self) -> List[Dict]:
        """Get all contacts from Monica."""
        return self._make_request('GET', 'contacts')['data']

    def get_contact_details(self, contact_id: int) -> Dict:
        """Get detailed information about a specific contact."""
        return self._make_request('GET', f'contacts/{contact_id}')['data']

    def get_conversations(self, contact_id: int) -> List[Dict]:
        """Get conversations for a specific contact."""
        return self._make_request(
            'GET', 
            'conversations',
            params={'contact_id': contact_id}
        )['data']

class MonicaRAG:
    def __init__(self, api_client: MonicaAPIClient, model_name: str = 'all-MiniLM-L6-v2'):
        self.api_client = api_client
        self.encoder = SentenceTransformer(model_name)
        self.storage = MonicaRAGStorage()
        self.embeddings = {}
        self.contact_data = {}

    def has_embeddings(self) -> bool:
        """Check if we have any embeddings stored."""
        with sqlite3.connect(self.storage.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            count = cursor.fetchone()[0]
            return count > 0
        
    def initialize(self):
        """Initialize the RAG system, loading or creating embeddings as needed."""
        if self.has_embeddings():
            print("Debug 1")
            self._load_from_storage()
            return True
        else:
            print("No existing embeddings found. Creating new ones...")
            self.update_embeddings()
            return False
        
    def _create_contact_text(self, contact: Dict) -> str:
        """Create a text representation of contact information."""
        parts = []
        
        # Basic information
        parts.append(f"Name: {contact.get('first_name', '')} {contact.get('last_name', '')}")
        if contact.get('job'):
            parts.append(f"Job: {contact['job']}")
        if contact.get('company'):
            parts.append(f"Company: {contact['company']}")
            
        # Notes and other information can be added here
        # You'll need to adjust based on Monica's actual API response structure
        
        return "\n".join(parts)
    
    def get_contacts(self):
        contacts = self.api_client.get_contacts()
        print(contacts)

    def update_embeddings(self):
        """Update embeddings for all contacts."""
        contacts = self.api_client.get_contacts()
        
        for contact in contacts:
            contact_id = contact['id']
            details = self.api_client.get_contact_details(contact_id)
            text = self._create_contact_text(details)
            embedding = self.encoder.encode(text)
            
            # Store in memory
            self.embeddings[contact_id] = embedding
            self.contact_data[contact_id] = details
            
            # Store in SQLite
            self.storage.save_embedding(contact_id, embedding, details)

    def _load_from_storage(self):
        """Load embeddings from SQLite."""
        stored_data = self.storage.get_all_embeddings()
        for contact_id, (embedding, contact_data) in stored_data.items():
            print(f"Loading embedding for contact {contact_id}, shape: {embedding.shape}")
            self.embeddings[contact_id] = embedding
            self.contact_data[contact_id] = contact_data

    def query(self, query: str, top_k: int = 3) -> List[Dict]:
        """Query contacts based on semantic similarity."""
        # Generate query embedding
        query_embedding = self.encoder.encode(query)
        
        # Calculate similarities
        similarities = {}
        for contact_id, embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities[contact_id] = similarity
            
        # Get top_k results
        top_results = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Return contact information for top results
        results = []
        for contact_id, similarity in top_results:
            results.append({
                'contact': self.contact_data[contact_id],
                'similarity': float(similarity)
            })
            
        return results

# Example usage:
if __name__ == "__main__":
    # Initialize clients
    api_client = MonicaAPIClient(
        base_url = base_url,
        token = token
    )

    rag = MonicaRAG(api_client)
    rag.initialize()
    
    # Example query
    query = "Who works in tech?"
    results = rag.query(query)
    print(query)
    for result in results:
        print(f"Contact: {result['contact']['first_name']} {result['contact']['last_name']}")
        print(f"Similarity: {result['similarity']:.3f}")
        print("---")