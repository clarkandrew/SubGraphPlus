import os
import argparse
import logging
from pathlib import Path
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError, ClientError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'migration.log'))
    ]
)
logger = logging.getLogger(__name__)

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
        
        # Execute the Cypher script in a transaction
        with driver.session() as session:
            tx = session.begin_transaction()
            
            try:
                # Run the migration script
                tx.run(cypher_script)
                
                # Record the migration version
                tx.run("""
                MERGE (s:_SchemaVersion {version: $version})
                SET s.appliedAt = timestamp()
                """, version=version)
                
                # Commit the transaction
                tx.commit()
                logger.info(f"Migration {version} applied successfully")
                return True
                
            except Exception as e:
                tx.rollback()
                logger.error(f"Migration {version} failed: {str(e)}")
                raise
                
    except FileNotFoundError:
        logger.error(f"Migration file not found: {file_path}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error reading/parsing migration file: {str(e)}")
        return False

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