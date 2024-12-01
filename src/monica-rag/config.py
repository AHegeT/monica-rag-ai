from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev')
    BASE_URL = os.getenv('MONICA_PROD_BASE_URL') if ENVIRONMENT == 'prod' else os.getenv('MONICA_DEV_BASE_URL')
    API_TOKEN = os.getenv('MONICA_PROD_API_TOKEN') if ENVIRONMENT == 'prod' else os.getenv('MONICA_DEV_API_TOKEN')
    DB_PATH = os.getenv('DB_PATH', 'monica_rag.db')
    MODEL_NAME = os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2')

config = Config()