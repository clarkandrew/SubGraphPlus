import os
import sqlite3
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from pathlib import Path
from app.config import config, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger(__name__)

# SQLite database path
SQLITE_DB_PATH = Path("data/staging.db")


class Neo4jDatabase:
    """Neo4j database connection manager"""
    
    def __init__(self):
        self._driver = None
        self._connect()
    
    def _connect(self):
        """Connect to Neo4j database"""
        try:
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
        if not self._driver:
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
        if not self._driver:
            self._connect()
        return self._driver.session()
    
    def run_query(self, query, params=None):
        """Run a Neo4j query"""
        with self.get_session() as session:
            result = session.run(query, params or {})
            return list(result)
    
    def run_transaction(self, tx_function, *args, **kwargs):
        """Run a transaction function"""
        with self.get_session() as session:
            return session.execute_write(tx_function, *args, **kwargs)


class SQLiteDatabase:
    """SQLite database connection manager"""
    
    def __init__(self):
        self.db_path = SQLITE_DB_PATH
        self._connection = None
        self._ensure_db_dir()
        self._connect()
        self._setup_schema()
    
    def _ensure_db_dir(self):
        """Ensure directory for SQLite DB exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _connect(self):
        """Connect to SQLite database"""
        try:
            self._connection = sqlite3.connect(str(self.db_path))
            self._connection.row_factory = sqlite3.Row
            logger.info(f"Connected to SQLite at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to SQLite: {str(e)}")
            self._connection = None
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
            attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT NOT NULL,
            is_correct BOOLEAN,
            comment TEXT,
            expected_answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        with self._connection:
            self._connection.executescript(schema_sql)
    
    def verify_connectivity(self):
        """Test SQLite connection"""
        if not self._connection:
            return False
        try:
            cursor = self._connection.cursor()
            cursor.execute("SELECT 1")
            return cursor.fetchone() is not None
        except sqlite3.Error:
            logger.error("SQLite is not available")
            return False
    
    def close(self):
        """Close the SQLite connection"""
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
        conn = self.get_connection()
        with conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            return cursor
    
    def executemany(self, query, params_list):
        """Execute a SQLite query with many parameter sets"""
        conn = self.get_connection()
        with conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            return cursor
    
    def fetchall(self, query, params=None):
        """Execute a query and fetch all results"""
        cursor = self.execute(query, params)
        return cursor.fetchall()
    
    def fetchone(self, query, params=None):
        """Execute a query and fetch one result"""
        cursor = self.execute(query, params)
        return cursor.fetchone()


# Create singleton instances
neo4j_db = Neo4jDatabase()
sqlite_db = SQLiteDatabase()


def close_connections():
    """Close all database connections"""
    neo4j_db.close()
    sqlite_db.close()