import os
import sqlite3
import logging
from pathlib import Path
import threading

# Add testing check first
TESTING = os.getenv('TESTING', '').lower() in ('1', 'true', 'yes')

# Conditional imports for neo4j
if not TESTING:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable
    from app.config import config, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
else:
    # Mock neo4j during testing
    GraphDatabase = None
    ServiceUnavailable = Exception
    from app.config import config
    NEO4J_URI = NEO4J_USER = NEO4J_PASSWORD = None

# RULE:import-rich-logger-correctly - Use centralized rich logger
from .log import logger, log_and_print
from rich.console import Console

# Initialize rich console for pretty CLI output
console = Console()

# Database paths
SQLITE_DB_PATH = config.sqlite_db_path if hasattr(config, 'sqlite_db_path') else "data/subgraph.db"


class Neo4jDatabase:
    """Neo4j database connection manager"""
    
    def __init__(self):
        self._driver = None
        if not TESTING:
            self._connect()
    
    def _connect(self):
        """Connect to Neo4j database"""
        if TESTING or GraphDatabase is None:
            logger.info("Skipping Neo4j connection in testing mode")
            return
            
        try:
            # For neo4j+s:// URIs, encryption is already specified in the URI scheme
            self._driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            # Test connection
            self.verify_connectivity()
            logger.info(f"Connected to Neo4j at {NEO4J_URI}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self._driver = None
            raise
    
    def verify_connectivity(self):
        """Test Neo4j connection"""
        if TESTING or not self._driver:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except ServiceUnavailable:
            logger.error("Neo4j is not available")
            return False
    
    def close(self):
        """Close the Neo4j connection"""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")
    
    def get_session(self):
        """Get a Neo4j session"""
        if TESTING:
            raise RuntimeError("Neo4j not available in testing mode")
        if not self._driver:
            self._connect()
        return self._driver.session()
    
    def run_query(self, query, params=None):
        """Run a Neo4j query"""
        if TESTING:
            return []
        with self.get_session() as session:
            result = session.run(query, params or {})
            return list(result)
    
    def run_transaction(self, tx_function, *args, **kwargs):
        """Run a transaction function"""
        if TESTING:
            return None
        with self.get_session() as session:
            return session.execute_write(tx_function, *args, **kwargs)


class SQLiteDatabase:
    """SQLite database connection manager"""
    
    def __init__(self):
        if TESTING:
            # Use in-memory database for testing
            self.db_path = ":memory:"
        else:
            self.db_path = SQLITE_DB_PATH
        self._connection = None
        self._lock = threading.Lock()
        if not TESTING:
            self._ensure_db_dir()
        self._connect()
        self._setup_schema()
    
    def _ensure_db_dir(self):
        """Ensure the database directory exists"""
        if not TESTING:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _connect(self):
        """Connect to SQLite database"""
        try:
            # Enable thread safety and WAL mode for better concurrency
            self._connection = sqlite3.connect(
                str(self.db_path), 
                check_same_thread=False,
                timeout=30.0
            )
            # Enable WAL mode for better concurrency (skip for in-memory)
            if not TESTING:
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute("PRAGMA synchronous=NORMAL")
            logger.info(f"Connected to SQLite database at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise
    
    def _setup_schema(self):
        """Set up SQLite schema"""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS staging_triples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            h_text TEXT NOT NULL,
            r_text TEXT NOT NULL,
            t_text TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP,
            UNIQUE(h_text, r_text, t_text)
        );
        CREATE INDEX IF NOT EXISTS idx_staging_status ON staging_triples(status);
        CREATE INDEX IF NOT EXISTS idx_staging_created_at ON staging_triples(created_at);
        
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash TEXT UNIQUE NOT NULL,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        );
        
        CREATE TABLE IF NOT EXISTS failed_auth_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT NOT NULL,
            is_correct BOOLEAN NOT NULL,
            comment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        with self._lock:
            self._connection.executescript(schema_sql)
    
    def verify_connectivity(self):
        """Test SQLite connection"""
        if not self._connection:
            return False
        try:
            with self._lock:
                cursor = self._connection.cursor()
                cursor.execute("SELECT 1")
                return cursor.fetchone() is not None
        except sqlite3.Error:
            logger.error("SQLite is not available")
            return False
    
    def close(self):
        """Close the SQLite connection"""
        with self._lock:
            if self._connection:
                self._connection.close()
                self._connection = None
                logger.info("SQLite connection closed")
    
    def get_connection(self):
        """Get SQLite connection"""
        if not self._connection:
            self._connect()
        return self._connection
    
    def execute(self, query, params=None):
        """Execute a SQLite query"""
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            conn.commit()
            return cursor
    
    def executemany(self, query, params_list):
        """Execute a SQLite query with many parameter sets"""
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor
    
    def fetchall(self, query, params=None):
        """Execute a query and fetch all results"""
        with self._lock:
            cursor = self.execute(query, params)
            return cursor.fetchall()
    
    def fetchone(self, query, params=None):
        """Execute a query and fetch one result"""
        with self._lock:
            cursor = self.execute(query, params)
            return cursor.fetchone()


# Create singleton instances with lazy initialization
# Skip database connections during testing to speed up tests and avoid connection errors
neo4j_db = None
sqlite_db = None

def get_neo4j_db():
    """Get Neo4j database instance with lazy initialization"""
    global neo4j_db
    if neo4j_db is None and not TESTING:
        logger.info("Initializing Neo4j connection...")
        neo4j_db = Neo4jDatabase()
    return neo4j_db

def get_sqlite_db():
    """Get SQLite database instance with lazy initialization"""
    global sqlite_db
    if sqlite_db is None and not TESTING:
        logger.info("Initializing SQLite connection...")
        sqlite_db = SQLiteDatabase()
    return sqlite_db

# Initialize databases immediately for now (can be made lazy later)
if not TESTING:
    # Initialize Neo4j database
    neo4j_db = Neo4jDatabase()
    sqlite_db = SQLiteDatabase()


def close_connections():
    """Close all database connections"""
    if neo4j_db:
        neo4j_db.close()
    if sqlite_db:
        sqlite_db.close()