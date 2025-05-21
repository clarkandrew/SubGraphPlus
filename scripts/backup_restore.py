import argparse
import os
import datetime
import shutil
import subprocess
import sys
import logging
import json
import time
from pathlib import Path

# Add parent directory to path so we can import app modules
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from app.config import DB_CONFIG
from app.database import sqlite_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backup_restore")

# Constants
BACKUP_DIR = os.path.join(parent_dir, "backups")
NEO4J_CONTAINER = "subgraphrag_neo4j"
NEO4J_BACKUP_CMD = "neo4j-admin dump"
NEO4J_RESTORE_CMD = "neo4j-admin load"
SQLITE_DB_PATH = os.path.join(parent_dir, "data", "staging.db")
FAISS_INDEX_DIR = os.path.join(parent_dir, "data", "faiss")


class BackupManager:
    """Manages backup and restoration of Neo4j, SQLite, and FAISS data"""

    def __init__(self, backup_dir=BACKUP_DIR):
        """Initialize backup manager"""
        self.backup_dir = backup_dir
        self._ensure_backup_dir()

    def _ensure_backup_dir(self):
        """Create backup directory if it doesn't exist"""
        os.makedirs(self.backup_dir, exist_ok=True)
        subdirs = ["neo4j", "sqlite", "faiss", "configs"]
        for subdir in subdirs:
            os.makedirs(os.path.join(self.backup_dir, subdir), exist_ok=True)

    def _generate_backup_id(self):
        """Generate backup ID based on timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"backup_{timestamp}"

    def _get_backup_metadata_path(self, backup_id):
        """Get path to backup metadata file"""
        return os.path.join(self.backup_dir, f"{backup_id}_metadata.json")

    def backup_neo4j(self, backup_id):
        """Backup Neo4j database using neo4j-admin dump"""
        try:
            output_file = os.path.join(self.backup_dir, "neo4j", f"{backup_id}.dump")

            # Check if Docker is running
            if shutil.which("docker"):
                try:
                    # Check if Neo4j container is running
                    check_container_cmd = [
                        "docker", "ps", "--format", "{{.Names}}",
                        "--filter", f"name={NEO4J_CONTAINER}"
                    ]
                    container_result = subprocess.run(
                        check_container_cmd,
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    
                    if NEO4J_CONTAINER not in container_result.stdout:
                        logger.warning(f"Neo4j container '{NEO4J_CONTAINER}' not found or not running.")
                        logger.info("Skipping Neo4j backup via Docker.")
                        return None
                        
                    logger.info("Backing up Neo4j via Docker...")
                    cmd = [
                        "docker", "exec", NEO4J_CONTAINER,
                        "bash", "-c", f"{NEO4J_BACKUP_CMD} --database=neo4j --to=/data/{backup_id}.dump"
                    ]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    
                    # Copy the dump file from the container
                    copy_cmd = [
                        "docker", "cp",
                        f"{NEO4J_CONTAINER}:/data/{backup_id}.dump",
                        output_file
                    ]
                    subprocess.run(copy_cmd, check=True)
                    
                    # Clean up the dump file in the container
                    cleanup_cmd = [
                        "docker", "exec", NEO4J_CONTAINER,
                        "rm", f"/data/{backup_id}.dump"
                    ]
                    subprocess.run(cleanup_cmd)
                except subprocess.SubprocessError as e:
                    logger.error(f"Docker Neo4j backup failed: {str(e)}")
                    logger.warning("Falling back to direct neo4j-admin backup...")
                    # Fall through to neo4j-admin backup
                else:
                    logger.info(f"Neo4j backup via Docker completed: {output_file}")
                    return output_file
                    
            # Try direct neo4j-admin if Docker failed or isn't available
            logger.warning("Using direct neo4j-admin backup...")
            if shutil.which("neo4j-admin"):
                # Direct backup if neo4j-admin is available
                cmd = [
                    "neo4j-admin", "dump",
                    "--database=neo4j",
                    f"--to={output_file}"
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"Neo4j backup via neo4j-admin completed: {output_file}")
                return output_file
            else:
                logger.error("Neither Docker nor neo4j-admin found. Cannot backup Neo4j.")
                logger.warning("Skipping Neo4j backup.")
                return None

        except subprocess.SubprocessError as e:
            logger.error(f"Neo4j backup failed: {str(e)}")
            logger.error(f"Command output: {e.stdout if hasattr(e, 'stdout') else ''}")
            logger.error(f"Command error: {e.stderr if hasattr(e, 'stderr') else ''}")
            logger.warning("Skipping Neo4j backup due to errors.")
            return None

    def backup_sqlite(self, backup_id):
        """Backup SQLite database"""
        try:
            output_file = os.path.join(self.backup_dir, "sqlite", f"{backup_id}.db")
            
            # Connect to SQLite and create a backup
            if os.path.exists(SQLITE_DB_PATH):
                # Use SQLite's backup command
                cmd = [
                    "sqlite3", SQLITE_DB_PATH,
                    f".backup '{output_file}'"
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Alternative: just copy the file
                if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                    shutil.copy2(SQLITE_DB_PATH, output_file)
                
                logger.info(f"SQLite backup completed: {output_file}")
                return output_file
            else:
                logger.warning(f"SQLite database not found at {SQLITE_DB_PATH}")
                return None
        except Exception as e:
            logger.error(f"SQLite backup failed: {str(e)}")
            raise RuntimeError(f"SQLite backup failed: {str(e)}")

    def backup_faiss(self, backup_id):
        """Backup FAISS index files"""
        try:
            output_dir = os.path.join(self.backup_dir, "faiss", backup_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # Copy FAISS index files
            if os.path.exists(FAISS_INDEX_DIR):
                for item in os.listdir(FAISS_INDEX_DIR):
                    source = os.path.join(FAISS_INDEX_DIR, item)
                    destination = os.path.join(output_dir, item)
                    
                    if os.path.isfile(source):
                        shutil.copy2(source, destination)
                    elif os.path.isdir(source):
                        shutil.copytree(source, destination)
                
                logger.info(f"FAISS backup completed: {output_dir}")
                return output_dir
            else:
                logger.warning(f"FAISS index directory not found at {FAISS_INDEX_DIR}")
                return None
        except Exception as e:
            logger.error(f"FAISS backup failed: {str(e)}")
            raise RuntimeError(f"FAISS backup failed: {str(e)}")

    def backup_configs(self, backup_id):
        """Backup configuration files"""
        try:
            output_dir = os.path.join(self.backup_dir, "configs", backup_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # List of config files to backup
            config_files = [
                os.path.join(parent_dir, "config", "app_config.json"),
                os.path.join(parent_dir, "config", "neo4j_schema.json"),
                os.path.join(parent_dir, ".env")
            ]
            
            # Copy each config file if it exists
            for config_file in config_files:
                if os.path.exists(config_file):
                    destination = os.path.join(output_dir, os.path.basename(config_file))
                    shutil.copy2(config_file, destination)
            
            logger.info(f"Config backup completed: {output_dir}")
            return output_dir
        except Exception as e:
            logger.error(f"Config backup failed: {str(e)}")
            raise RuntimeError(f"Config backup failed: {str(e)}")

    def create_backup(self):
        """Create full backup of all components"""
        backup_id = self._generate_backup_id()
        logger.info(f"Starting backup {backup_id}...")
        
        backup_paths = {}
        errors = []
        
        # Backup each component
        try:
            backup_paths["neo4j"] = self.backup_neo4j(backup_id)
        except Exception as e:
            errors.append(f"Neo4j: {str(e)}")
            logger.error(f"Neo4j backup failed: {str(e)}")
        
        try:
            backup_paths["sqlite"] = self.backup_sqlite(backup_id)
        except Exception as e:
            errors.append(f"SQLite: {str(e)}")
            logger.error(f"SQLite backup failed: {str(e)}")
        
        try:
            backup_paths["faiss"] = self.backup_faiss(backup_id)
        except Exception as e:
            errors.append(f"FAISS: {str(e)}")
            logger.error(f"FAISS backup failed: {str(e)}")
        
        try:
            backup_paths["configs"] = self.backup_configs(backup_id)
        except Exception as e:
            errors.append(f"Configs: {str(e)}")
            logger.error(f"Config backup failed: {str(e)}")
        
        # Create metadata file
        metadata = {
            "backup_id": backup_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "paths": backup_paths,
            "errors": errors,
            "success": len(errors) == 0
        }
        
        with open(self._get_backup_metadata_path(backup_id), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Backup {backup_id} completed with {len(errors)} errors")
        return backup_id, metadata

    def restore_neo4j(self, backup_path):
        """Restore Neo4j database from backup"""
        try:
            if not os.path.exists(backup_path):
                logger.error(f"Neo4j backup file not found: {backup_path}")
                return False
            
            # Check if Docker is running
            if shutil.which("docker"):
                try:
                    # Check if Neo4j container is running
                    check_container_cmd = [
                        "docker", "ps", "--format", "{{.Names}}",
                        "--filter", f"name={NEO4J_CONTAINER}"
                    ]
                    container_result = subprocess.run(
                        check_container_cmd,
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    
                    if NEO4J_CONTAINER not in container_result.stdout:
                        logger.warning(f"Neo4j container '{NEO4J_CONTAINER}' not found or not running.")
                        logger.warning("Skipping Neo4j restore via Docker.")
                    else:
                        logger.info("Restoring Neo4j via Docker...")
                        
                        # Copy the dump file to the container
                        copy_cmd = [
                            "docker", "cp",
                            backup_path,
                            f"{NEO4J_CONTAINER}:/data/restore.dump"
                        ]
                        subprocess.run(copy_cmd, check=True)
                        
                        # Stop Neo4j service
                        stop_cmd = [
                            "docker", "exec", NEO4J_CONTAINER,
                            "neo4j", "stop"
                        ]
                        subprocess.run(stop_cmd, check=True)
                        
                        # Restore the database
                        cmd = [
                            "docker", "exec", NEO4J_CONTAINER,
                            "bash", "-c", f"{NEO4J_RESTORE_CMD} --database=neo4j --from=/data/restore.dump --force"
                        ]
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        # Start Neo4j service
                        start_cmd = [
                            "docker", "exec", NEO4J_CONTAINER,
                            "neo4j", "start"
                        ]
                        subprocess.run(start_cmd, check=True)
                        
                        # Clean up the dump file in the container
                        cleanup_cmd = [
                            "docker", "exec", NEO4J_CONTAINER,
                            "rm", "/data/restore.dump"
                        ]
                        subprocess.run(cleanup_cmd)
                        
                        logger.info("Neo4j restore via Docker completed")
                        return True
                except subprocess.SubprocessError as e:
                    logger.error(f"Docker Neo4j restore failed: {str(e)}")
                    logger.warning("Falling back to direct neo4j-admin restore...")
                    # Fall through to neo4j-admin restore
            
            # Try direct neo4j-admin if Docker failed or isn't available
            logger.warning("Using direct neo4j-admin restore...")
            if shutil.which("neo4j-admin"):
                try:
                    # Stop Neo4j service
                    subprocess.run(["neo4j", "stop"], check=True)
                    
                    # Direct restore if neo4j-admin is available
                    cmd = [
                        "neo4j-admin", "load",
                        "--database=neo4j",
                        f"--from={backup_path}",
                        "--force"
                    ]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    
                    # Start Neo4j service
                    subprocess.run(["neo4j", "start"], check=True)
                    
                    logger.info("Neo4j restore via neo4j-admin completed")
                    return True
                except subprocess.SubprocessError as e:
                    logger.error(f"Neo4j-admin restore failed: {str(e)}")
                    logger.error(f"Command output: {e.stdout if hasattr(e, 'stdout') else ''}")
                    logger.error(f"Command error: {e.stderr if hasattr(e, 'stderr') else ''}")
                    return False
            else:
                logger.error("Neither Docker nor neo4j-admin found. Cannot restore Neo4j.")
                return False

        except Exception as e:
            logger.error(f"Neo4j restore failed: {str(e)}")
            return False

    def restore_sqlite(self, backup_path):
        """Restore SQLite database from backup"""
        try:
            if not os.path.exists(backup_path):
                raise FileNotFoundError(f"SQLite backup file not found: {backup_path}")
            
            # Make sure the data directory exists
            os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
            
            # Create backup of existing database if it exists
            if os.path.exists(SQLITE_DB_PATH):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{SQLITE_DB_PATH}.{timestamp}.bak"
                shutil.copy2(SQLITE_DB_PATH, backup_name)
                logger.info(f"Created backup of existing SQLite database: {backup_name}")
            
            # Copy backup file to database location
            shutil.copy2(backup_path, SQLITE_DB_PATH)
            
            logger.info("SQLite restore completed")
            return True
        except Exception as e:
            logger.error(f"SQLite restore failed: {str(e)}")
            raise RuntimeError(f"SQLite restore failed: {str(e)}")

    def restore_faiss(self, backup_dir):
        """Restore FAISS index files from backup"""
        try:
            if not os.path.exists(backup_dir):
                raise FileNotFoundError(f"FAISS backup directory not found: {backup_dir}")
            
            # Make sure the FAISS directory exists
            os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
            
            # Create backup of existing FAISS directory if it exists
            if os.path.exists(FAISS_INDEX_DIR) and os.listdir(FAISS_INDEX_DIR):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{FAISS_INDEX_DIR}.{timestamp}.bak"
                shutil.copytree(FAISS_INDEX_DIR, backup_name)
                logger.info(f"Created backup of existing FAISS directory: {backup_name}")
            
            # Clear existing directory
            for item in os.listdir(FAISS_INDEX_DIR):
                item_path = os.path.join(FAISS_INDEX_DIR, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            
            # Copy backup files to FAISS directory
            for item in os.listdir(backup_dir):
                source = os.path.join(backup_dir, item)
                destination = os.path.join(FAISS_INDEX_DIR, item)
                
                if os.path.isfile(source):
                    shutil.copy2(source, destination)
                elif os.path.isdir(source):
                    shutil.copytree(source, destination)
            
            logger.info("FAISS restore completed")
            return True
        except Exception as e:
            logger.error(f"FAISS restore failed: {str(e)}")
            raise RuntimeError(f"FAISS restore failed: {str(e)}")

    def restore_configs(self, backup_dir):
        """Restore configuration files from backup"""
        try:
            if not os.path.exists(backup_dir):
                raise FileNotFoundError(f"Config backup directory not found: {backup_dir}")
            
            # List of config files to restore
            config_paths = {
                "app_config.json": os.path.join(parent_dir, "config", "app_config.json"),
                "neo4j_schema.json": os.path.join(parent_dir, "config", "neo4j_schema.json"),
                ".env": os.path.join(parent_dir, ".env")
            }
            
            # Restore each config file if it exists in backup
            for config_name, dest_path in config_paths.items():
                source_path = os.path.join(backup_dir, config_name)
                
                if os.path.exists(source_path):
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    # Backup existing config if it exists
                    if os.path.exists(dest_path):
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_name = f"{dest_path}.{timestamp}.bak"
                        shutil.copy2(dest_path, backup_name)
                    
                    # Copy backup config to destination
                    shutil.copy2(source_path, dest_path)
            
            logger.info("Config restore completed")
            return True
        except Exception as e:
            logger.error(f"Config restore failed: {str(e)}")
            raise RuntimeError(f"Config restore failed: {str(e)}")

    def get_available_backups(self):
        """Get list of available backups"""
        backups = []
        
        # Find all metadata files
        for filename in os.listdir(self.backup_dir):
            if filename.startswith("backup_") and filename.endswith("_metadata.json"):
                metadata_path = os.path.join(self.backup_dir, filename)
                
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        backups.append(metadata)
                except Exception as e:
                    logger.warning(f"Could not read metadata file {metadata_path}: {str(e)}")
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return backups

    def restore_backup(self, backup_id=None):
        """Restore from backup"""
        # If backup_id not provided, use latest backup
        if not backup_id:
            backups = self.get_available_backups()
            if not backups:
                logger.error("No backups found")
                return False
            backup_id = backups[0]["backup_id"]
        
        # Load backup metadata
        metadata_path = self._get_backup_metadata_path(backup_id)
        if not os.path.exists(metadata_path):
            logger.error(f"Backup metadata not found: {metadata_path}")
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load backup metadata: {str(e)}")
            return False
        
        logger.info(f"Starting restore of backup {backup_id}...")
        
        errors = []
        success_components = []
        
        # Restore Neo4j
        if "neo4j" in metadata["paths"] and metadata["paths"]["neo4j"]:
            if self.restore_neo4j(metadata["paths"]["neo4j"]):
                success_components.append("Neo4j")
            else:
                errors.append("Neo4j: Failed to restore")
        
        # Restore SQLite
        if "sqlite" in metadata["paths"] and metadata["paths"]["sqlite"]:
            try:
                if self.restore_sqlite(metadata["paths"]["sqlite"]):
                    success_components.append("SQLite")
                else:
                    errors.append("SQLite: Failed to restore")
            except Exception as e:
                errors.append(f"SQLite: {str(e)}")
        
        # Restore FAISS
        if "faiss" in metadata["paths"] and metadata["paths"]["faiss"]:
            try:
                if self.restore_faiss(metadata["paths"]["faiss"]):
                    success_components.append("FAISS")
                else:
                    errors.append("FAISS: Failed to restore")
            except Exception as e:
                errors.append(f"FAISS: {str(e)}")
        
        # Restore configs
        if "configs" in metadata["paths"] and metadata["paths"]["configs"]:
            try:
                if self.restore_configs(metadata["paths"]["configs"]):
                    success_components.append("Configs")
                else:
                    errors.append("Configs: Failed to restore")
            except Exception as e:
                errors.append(f"Configs: {str(e)}")
        
        logger.info(f"Restore of backup {backup_id} completed with {len(errors)} errors")
        logger.info(f"Successfully restored components: {', '.join(success_components)}")
        
        if errors:
            for error in errors:
                logger.error(f"Restore error: {error}")
        
        return len(errors) == 0


def main():
    """Main entry point for backup_restore script"""
    parser = argparse.ArgumentParser(description="SubgraphRAG+ Backup and Restore")
    parser.add_argument("--action", choices=["backup", "restore", "list"], required=True, 
                        help="Action to perform: backup, restore, or list")
    parser.add_argument("--backup-id", help="Backup ID for restore (default: latest)")
    parser.add_argument("--backup-dir", default=BACKUP_DIR, help="Backup directory")
    parser.add_argument("--skip-docker-check", action="store_true", 
                        help="Skip Docker container availability check")
    
    args = parser.parse_args()
    
    # Ensure backup directory exists
    os.makedirs(args.backup_dir, exist_ok=True)
    
    manager = BackupManager(args.backup_dir)
    
    try:
        if args.action == "backup":
            backup_id, metadata = manager.create_backup()
            print(f"Backup {backup_id} created:")
            print(f"  - Neo4j: {'✓' if 'neo4j' in metadata['paths'] and metadata['paths']['neo4j'] else '✗'}")
            print(f"  - SQLite: {'✓' if 'sqlite' in metadata['paths'] and metadata['paths']['sqlite'] else '✗'}")
            print(f"  - FAISS: {'✓' if 'faiss' in metadata['paths'] and metadata['paths']['faiss'] else '✗'}")
            print(f"  - Configs: {'✓' if 'configs' in metadata['paths'] and metadata['paths']['configs'] else '✗'}")
            
            # Count successful components
            successful = sum(1 for component in ['neo4j', 'sqlite', 'faiss', 'configs'] 
                            if component in metadata['paths'] and metadata['paths'][component])
            print(f"Successfully backed up {successful}/4 components")
            
            if metadata.get("errors"):
                print("Errors occurred during backup:")
                for error in metadata["errors"]:
                    print(f"  - {error}")
            
        elif args.action == "restore":
            success = manager.restore_backup(args.backup_id)
            if success:
                print("✓ Restore completed successfully")
            else:
                print("⚠ Restore completed with some issues")
                print("  Check the logs for more details")
            
        elif args.action == "list":
            backups = manager.get_available_backups()
            if not backups:
                print("No backups found in", args.backup_dir)
            else:
                print(f"Found {len(backups)} backups:")
                for i, backup in enumerate(backups):
                    status = "✓" if backup.get("success") else "⚠"
                    timestamp = backup.get("timestamp", "unknown")
                    backup_id = backup.get("backup_id", "unknown")
                    
                    # Count components
                    components = []
                    if backup.get("paths", {}).get("neo4j"):
                        components.append("Neo4j")
                    if backup.get("paths", {}).get("sqlite"):
                        components.append("SQLite")
                    if backup.get("paths", {}).get("faiss"):
                        components.append("FAISS")
                    if backup.get("paths", {}).get("configs"):
                        components.append("Configs")
                    
                    print(f"{i+1}. [{status}] {backup_id} - {timestamp}")
                    print(f"   Components: {', '.join(components) if components else 'None'}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())