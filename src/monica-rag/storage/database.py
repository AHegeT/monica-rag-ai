import sqlite3
import json
import numpy as np
from typing import Dict
from datetime import datetime

class MonicaRAGStorage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self) -> None:
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

    def save_embedding(self, contact_id: int, embedding: np.ndarray, contact_data: Dict) -> None:
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
            return {
                contact_id: (
                    np.frombuffer(embedding_bytes, dtype=np.float32).copy(),
                    json.loads(contact_data)
                )
                for contact_id, embedding_bytes, contact_data in cursor
            }

    def get_last_updated(self) -> datetime:
        """Get the timestamp of the most recent update."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT MAX(last_updated) FROM embeddings")
            return cursor.fetchone()[0]