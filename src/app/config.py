import os
import json
import logging
import jsonschema
from pathlib import Path

# Load environment variables from .env file before accessing them
from dotenv import load_dotenv
load_dotenv()

# Add this at the top after imports, before any expensive operations
TESTING = os.getenv('TESTING', '').lower() in ('1', 'true', 'yes')
DISABLE_MODELS = os.getenv('DISABLE_MODELS', '').lower() in ('1', 'true', 'yes')

# If testing, skip expensive operations
if TESTING:
    logging.basicConfig(level=logging.CRITICAL)  # Reduce logging noise

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'app.log'))
    ]
)
logger = logging.getLogger(__name__)

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up from src/app/ to project root
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, CACHE_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Environment variables
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
API_KEY_SECRET = os.environ.get("API_KEY_SECRET", "default_key_for_dev_only")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# MLX Configuration (for LLM only when MODEL_BACKEND=mlx)
MLX_LLM_MODEL = os.environ.get("MLX_LLM_MODEL", "mlx-community/Mistral-7B-Instruct-v0.2-4bit-mlx")
MLX_LLM_MODEL_PATH = os.environ.get("MLX_LLM_MODEL_PATH")

# Embedding model (always uses HuggingFace transformers regardless of MODEL_BACKEND)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "Alibaba-NLP/gte-large-en-v1.5")

# Configuration class
class Config:
    def __init__(self):
        self.schema_path = CONFIG_DIR / "config.schema.json"
        self.config_path = CONFIG_DIR / "config.json"
        self.aliases_path = CONFIG_DIR / "aliases.json"
        
        self.config_data = {}
        self.aliases = {}
        
        self._load_schema()
        self._load_config()
        self._load_aliases()

    def _load_schema(self):
        """Load and parse the configuration schema"""
        try:
            with open(self.schema_path, 'r') as f:
                self.schema = json.load(f)
                logger.info(f"Loaded schema from {self.schema_path}")
        except FileNotFoundError:
            logger.critical(f"Schema file not found: {self.schema_path}")
            raise
        except json.JSONDecodeError as e:
            logger.critical(f"Invalid schema JSON: {e}")
            raise

    def _load_config(self):
        """Load and validate configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)
                logger.info(f"Loaded config from {self.config_path}")
        except FileNotFoundError:
            logger.critical(f"Config file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.critical(f"Invalid config JSON: {e}")
            raise
        
        # Validate against schema
        try:
            jsonschema.validate(instance=self.config_data, schema=self.schema)
            logger.info("Config validation successful")
        except jsonschema.exceptions.ValidationError as e:
            logger.critical(f"Config validation failed: {e}")
            raise

        # Set log level after loading config
        log_level = self.config_data.get("LOG_LEVEL", "INFO")
        numeric_level = getattr(logging, log_level, None)
        if isinstance(numeric_level, int):
            logging.getLogger().setLevel(numeric_level)
            logger.info(f"Set log level to {log_level}")
        
    def _load_aliases(self):
        """Load aliases for entity linking"""
        try:
            with open(self.aliases_path, 'r') as f:
                self.aliases = json.load(f)
                logger.info(f"Loaded {len(self.aliases)} alias entries")
        except FileNotFoundError:
            logger.warning(f"Aliases file not found: {self.aliases_path}")
            self.aliases = {}
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid aliases JSON: {e}")
            self.aliases = {}

    def __getattr__(self, name):
        """Access config values as attributes"""
        if name in self.config_data:
            return self.config_data[name]
        raise AttributeError(f"Config has no attribute '{name}'")

# Create singleton instance
config = Config()

# Path helpers
def get_path(relative_path):
    """Convert a relative path to absolute project path"""
    return PROJECT_ROOT / relative_path

def get_cache_path(cache_subdir=None):
    """Get path to cache directory or subdirectory"""
    path = CACHE_DIR
    if cache_subdir:
        path = path / cache_subdir
        path.mkdir(exist_ok=True, parents=True)
    return path