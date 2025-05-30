import os
import argparse
import sys
import time
from pathlib import Path
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError, ClientError

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# RULE:import-rich-logger-correctly - Use centralized rich logger
from src.app.log import logger, log_and_print
from rich.console import Console

# Initialize rich console for pretty CLI output
console = Console()

def get_neo4j_connection():
    """Get a Neo4j connection from environment variables"""
    neo4j_uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")
    
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
        return driver
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {str(e)}")
        raise

def get_applied_versions(driver):
    """Get already applied schema versions"""
    with driver.session() as session:
        result = session.run("""
        MATCH (s:_SchemaVersion)
        RETURN s.version AS version, s.appliedAt AS appliedAt
        ORDER BY s.appliedAt
        """)
        return {record["version"]: record["appliedAt"] for record in result}

def apply_migration(driver, file_path, version):
    """Apply a migration file to Neo4j"""
    logger.info(f"Applying migration: {file_path}")
    
    try:
        with open(file_path, 'r') as file:
            cypher_script = file.read()
        
        # Split the script into individual statements
        statements = parse_cypher_statements(cypher_script)
        
        # Separate schema and data statements
        schema_statements = []
        data_statements = []
        
        for statement in statements:
            statement_upper = statement.strip().upper()
            if (statement_upper.startswith('CREATE CONSTRAINT') or 
                statement_upper.startswith('CREATE INDEX') or
                statement_upper.startswith('DROP CONSTRAINT') or
                statement_upper.startswith('DROP INDEX') or
                statement_upper.startswith('CALL DB.INDEX')):
                schema_statements.append(statement)
            else:
                data_statements.append(statement)
        
        # Execute schema modifications first (each in its own transaction)
        for i, statement in enumerate(schema_statements):
            if statement.strip():
                logger.debug(f"Executing schema statement {i+1}: {statement[:100]}...")
                with driver.session() as session:
                    session.run(statement)
        
        # Execute data modifications in a single transaction
        if data_statements:
            with driver.session() as session:
                tx = session.begin_transaction()
                
                try:
                    for i, statement in enumerate(data_statements):
                        if statement.strip():
                            logger.debug(f"Executing data statement {i+1}: {statement[:100]}...")
                            tx.run(statement)
                    
                    # Record the migration version
                    tx.run("""
                    MERGE (s:_SchemaVersion {version: $version})
                    SET s.appliedAt = timestamp()
                    """, version=version)
                    
                    # Commit the transaction
                    tx.commit()
                    
                except Exception as e:
                    tx.rollback()
                    logger.error(f"Migration {version} failed: {str(e)}")
                    raise
        else:
            # If no data statements, just record the migration version
            with driver.session() as session:
                session.run("""
                MERGE (s:_SchemaVersion {version: $version})
                SET s.appliedAt = timestamp()
                """, version=version)
        
        logger.info(f"Migration {version} applied successfully")
        return True
                
    except FileNotFoundError:
        logger.error(f"Migration file not found: {file_path}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error reading/parsing migration file: {str(e)}")
        return False

def parse_cypher_statements(cypher_script):
    """
    Parse a Cypher script into individual statements
    
    This function splits a multi-statement Cypher script into individual
    statements that can be executed separately.
    
    Args:
        cypher_script: String containing the full Cypher script
        
    Returns:
        List of individual Cypher statements
    """
    # Remove comments and split by semicolons
    lines = []
    for line in cypher_script.split('\n'):
        line = line.strip()
        # Skip comment lines
        if line.startswith('//') or not line:
            continue
        # Remove inline comments
        if '//' in line:
            line = line[:line.index('//')]
        lines.append(line)
    
    # Join lines and split by semicolons
    clean_script = ' '.join(lines)
    statements = [stmt.strip() for stmt in clean_script.split(';') if stmt.strip()]
    
    return statements

def discover_migrations(migrations_dir):
    """Discover migration files in the migrations directory"""
    migrations_path = Path(migrations_dir)
    if not migrations_path.exists():
        logger.warning(f"Migrations directory not found: {migrations_dir}")
        return {}
    
    migrations = {}
    for file_path in migrations_path.glob("*.cypher"):
        # Extract version from filename: NNN_description.cypher -> vNNN
        file_name = file_path.stem
        parts = file_name.split('_', 1)
        if len(parts) == 2 and parts[0].isdigit():
            version = f"kg_v{parts[0]}"
            migrations[version] = file_path
    
    return migrations

def run_migrations(target_version=None):
    """Run schema migrations up to the target version"""
    driver = get_neo4j_connection()
    migrations_dir = "migrations/neo4j"
    
    # Get already applied versions
    applied_versions = get_applied_versions(driver)
    logger.info(f"Found {len(applied_versions)} already applied migrations: {list(applied_versions.keys())}")
    
    # Discover available migrations
    available_migrations = discover_migrations(migrations_dir)
    logger.info(f"Found {len(available_migrations)} available migrations: {list(available_migrations.keys())}")
    
    # Sort migrations by version number
    sorted_migrations = sorted(available_migrations.keys(), key=lambda v: int(v.split('v')[1]))
    
    # Determine which migrations to apply
    migrations_to_apply = []
    for version in sorted_migrations:
        if version in applied_versions:
            logger.info(f"Skipping already applied migration: {version}")
            continue
        
        if target_version and version > target_version:
            logger.info(f"Skipping migration {version} (target is {target_version})")
            break
        
        migrations_to_apply.append((version, available_migrations[version]))
    
    # Apply migrations in order
    if not migrations_to_apply:
        logger.info("No migrations to apply.")
        return
    
    logger.info(f"Applying {len(migrations_to_apply)} migrations: {[m[0] for m in migrations_to_apply]}")
    
    for version, file_path in migrations_to_apply:
        success = apply_migration(driver, file_path, version)
        if not success:
            logger.error(f"Migration {version} failed, stopping.")
            break
    
    driver.close()
    logger.info("Migration process completed")

if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(description="Neo4j Schema Migration Tool")
    
    # Add arguments
    parser.add_argument(
        "--target-version", 
        type=str, 
        help="Target schema version (e.g. kg_v3)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run migrations up to target version
    run_migrations(args.target_version)