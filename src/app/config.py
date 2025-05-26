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

# RULE:import-rich-logger-correctly - Use centralized rich logger
from .log import logger, log_and_print
from rich.console import Console

# Initialize rich console for pretty CLI output
console = Console()

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

# Environment variables (secrets and environment-specific values)
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
API_KEY_SECRET = os.environ.get("API_KEY_SECRET", "default_key_for_dev_only")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

# Optional custom paths from environment
MLX_LLM_MODEL_PATH = os.environ.get("MLX_LLM_MODEL_PATH")
HF_MODEL_PATH = os.environ.get("HF_MODEL_PATH")

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
        self._setup_backward_compatibility()

    def _load_schema(self):
        """Load and parse the configuration schema"""
        try:
            with open(self.schema_path, 'r') as f:
                self.schema = json.load(f)
                logger.info(f"Loaded schema from {self.schema_path}")
        except FileNotFoundError:
            logger.warning(f"Schema file not found: {self.schema_path}, skipping validation")
            self.schema = None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid schema JSON: {e}")
            self.schema = None

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
        
        # Validate against schema if available
        if self.schema:
            try:
                jsonschema.validate(instance=self.config_data, schema=self.schema)
                logger.info("Config validation successful")
            except jsonschema.exceptions.ValidationError as e:
                logger.warning(f"Config validation failed: {e}")
        
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

    def _setup_backward_compatibility(self):
        """Setup backward compatibility attributes for legacy code"""
        # Model backend
        self.MODEL_BACKEND = self.config_data.get("models", {}).get("backend", "mlx")
        
        # LLM Models
        llm_config = self.config_data.get("models", {}).get("llm", {})
        mlx_config = llm_config.get("mlx", {})
        
        # MLX LLM Model (use environment override if available)
        self.MLX_LLM_MODEL = MLX_LLM_MODEL_PATH or mlx_config.get("model", "mlx-community/Qwen3-14B-8bit")
        
        # OpenAI config
        openai_config = llm_config.get("openai", {})
        self.OPENAI_MODEL = openai_config.get("model", "gpt-3.5-turbo")
        
        # HuggingFace config
        hf_config = llm_config.get("huggingface", {})
        self.HF_MODEL = HF_MODEL_PATH or hf_config.get("model", "mistralai/Mistral-7B-Instruct-v0.2")
        
        # Embedding Model (ALWAYS uses transformers, never MLX)
        embedding_config = self.config_data.get("models", {}).get("embeddings", {})
        self.EMBEDDING_MODEL = embedding_config.get("model", "Alibaba-NLP/gte-large-en-v1.5")
        
        # Paths
        self.FAISS_INDEX_PATH = self.config_data.get("data", {}).get("faiss_index_path", "data/faiss_index.bin")
        self.MLP_MODEL_PATH = self.config_data.get("models", {}).get("mlp", {}).get("model_path", "models/mlp/mlp.pth")
        
        # Performance settings
        perf_config = self.config_data.get("performance", {})
        self.TOKEN_BUDGET = self.config_data.get("retrieval", {}).get("token_budget", 4000)
        self.MAX_DDE_HOPS = self.config_data.get("retrieval", {}).get("max_dde_hops", 2)
        self.CACHE_SIZE = perf_config.get("cache_size", 1000)
        self.INGEST_BATCH_SIZE = perf_config.get("ingest_batch_size", 100)
        self.API_RATE_LIMIT = perf_config.get("api_rate_limit", 60)
        
        # Directories
        paths_config = self.config_data.get("paths", {})
        self.CACHE_DIR = paths_config.get("cache_dir", "cache/")
        
        # Legacy LOG_LEVEL support
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    def get_model_config(self, backend: str = None) -> dict:
        """Get model configuration for a specific backend"""
        backend = backend or self.MODEL_BACKEND
        return self.config_data.get("models", {}).get("llm", {}).get(backend, {})

    def get_embedding_config(self) -> dict:
        """Get embedding model configuration"""
        return self.config_data.get("models", {}).get("embeddings", {})

    def get_retrieval_config(self) -> dict:
        """Get retrieval configuration"""
        return self.config_data.get("retrieval", {})

    def get_performance_config(self) -> dict:
        """Get performance configuration"""
        return self.config_data.get("performance", {})

    def get_information_extraction_config(self) -> dict:
        """Get information extraction model configuration"""
        return self.config_data.get("models", {}).get("information_extraction", {})
    
    def get_rebel_config(self) -> dict:
        """Get REBEL relation extraction model configuration"""
        return self.get_information_extraction_config().get("rebel", {})
    
    def get_entity_typing_config(self) -> dict:
        """Get entity typing model configuration"""
        return self.config_data.get("models", {}).get("entity_typing", {})
    
    def get_ontonotes_config(self) -> dict:
        """Get OntoNotes NER model configuration"""
        return self.get_entity_typing_config().get("ontonotes_ner", {})
    
    def get_spacy_config(self) -> dict:
        """Get spaCy fallback model configuration"""
        return self.get_entity_typing_config().get("spacy_fallback", {})

    def __getattr__(self, name):
        """Access config values as attributes with nested support"""
        # Try direct access first
        if name in self.config_data:
            return self.config_data[name]
        
        # Try nested access (e.g., models.backend)
        parts = name.lower().split('_')
        current = self.config_data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                break
        else:
            return current
        
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

# Export commonly used values for convenience
MODEL_BACKEND = config.MODEL_BACKEND
MLX_LLM_MODEL = config.MLX_LLM_MODEL
EMBEDDING_MODEL = config.EMBEDDING_MODEL
FAISS_INDEX_PATH = config.FAISS_INDEX_PATH
MLP_MODEL_PATH = config.MLP_MODEL_PATH