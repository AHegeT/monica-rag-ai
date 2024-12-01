import logging
from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer

from ..api.client import MonicaAPIClient
from ..storage.database import MonicaRAGStorage
from ..utils.text_processing import create_contact_text
from ..config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonicaRAG:
    def __init__(
        self,
        api_client: MonicaAPIClient,
        model_name: str = None,
        db_path: str = None
    ):
        """
        Initialize the RAG system.
        
        Args:
            api_client: Initialized MonicaAPIClient
            model_name: Name of the sentence transformer model to use
            db_path: Path to the SQLite database
        """
        self.api_client = api_client
        self.encoder = SentenceTransformer(model_name or config.MODEL_NAME)
        self.storage = MonicaRAGStorage(db_path or config.DB_PATH)
        self.embeddings = {}
        self.contact_data = {}

    def initialize(self) -> bool:
        """
        Initialize the RAG system, loading or creating embeddings as needed.
        
        Returns:
            bool: True if embeddings were loaded from storage, False if new ones were created
        """
        try:
            if self._has_embeddings():
                logger.info("Loading existing embeddings from storage...")
                self._load_from_storage()
                return True
            else:
                logger.info("No existing embeddings found. Creating new ones...")
                self.update_embeddings()
                return False
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            raise

    def _has_embeddings(self) -> bool:
        """Check if we have any embeddings stored."""
        try:
            with self.storage.db_path as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                count = cursor.fetchone()[0]
                return count > 0
        except Exception as e:
            logger.error(f"Error checking embeddings: {str(e)}")
            return False

    def update_embeddings(self) -> None:
        """Update embeddings for all contacts."""
        try:
            contacts = self.api_client.get_contacts()
            logger.info(f"Updating embeddings for {len(contacts)} contacts...")
            
            for contact in contacts:
                contact_id = contact['id']
                details = self.api_client.get_contact_details(contact_id)
                text = create_contact_text(details)
                embedding = self.encoder.encode(text)
                
                # Store in memory
                self.embeddings[contact_id] = embedding
                self.contact_data[contact_id] = details
                
                # Store in SQLite
                self.storage.save_embedding(contact_id, embedding, details)
                
            logger.info("Successfully updated embeddings")
        except Exception as e:
            logger.error(f"Failed to update embeddings: {str(e)}")
            raise

    def _load_from_storage(self) -> None:
        """Load embeddings from SQLite."""
        try:
            stored_data = self.storage.get_all_embeddings()
            for contact_id, (embedding, contact_data) in stored_data.items():
                logger.debug(f"Loading embedding for contact {contact_id}")
                self.embeddings[contact_id] = embedding
                self.contact_data[contact_id] = contact_data
            logger.info(f"Loaded {len(stored_data)} embeddings from storage")
        except Exception as e:
            logger.error(f"Failed to load embeddings from storage: {str(e)}")
            raise

    def query(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Query contacts based on semantic similarity.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing contact information and similarity scores
        """
        try:
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
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            raise