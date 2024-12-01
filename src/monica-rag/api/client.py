from ssl import SSLError
import requests
from typing import Dict, List
import urllib3

class MonicaAPIClient:
    def __init__(self, base_url: str, token: str, verify_ssl: bool = True):
        """Initialize the Monica API client"""
        self.base_url = self._normalize_base_url(base_url)
        self.token = token
        self.verify_ssl = verify_ssl
        self.session = self._setup_session()

        if not verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """Normalize the base URL format."""
        base_url = base_url.rstrip('/')
        if not base_url.startswith('https://'):
            base_url = f'https://{base_url}'
        return base_url

    def _setup_session(self) -> requests.Session:
        """Set up and configure requests session."""
        session = requests.Session()
        session.headers.update({
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        })
        return session

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
                "SSL verification failed. If using a self-signed certificate, "
                "set verify_ssl=False (development only)."
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