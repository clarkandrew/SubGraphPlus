import os
import argparse
import csv
import logging
import sqlite3
from pathlib import Path
import sys

# Add parent directory to path so we can import app modules
sys.path.append(str(Path(__file__).parent.parent))
from app.database import sqlite_db

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'ingest.log'))
    ]
)
logger = logging.getLogger(__name__)

def stage_triples_from_csv(file_path):
    """
    Load triples from CSV file into SQLite staging table.
    CSV should have format: head,relation,tail
    """
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return 0
    
    # Prepare counters
    total_rows = 0
    successful = 0
    duplicates = 0
    errors = 0
    
    try:
        # Read CSV file
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check for required columns
            required_columns = ['head', 'relation', 'tail']
            for col in required_columns:
                if col not in reader.fieldnames:
                    logger.error(f"CSV file missing required column: {col}")
                    return 0
            
            # Process each row
            for row in reader:
                total_rows += 1
                try:
                    # Extract triple
                    head = row['head'].strip()
                    relation = row['relation'].strip()
                    tail = row['tail'].strip()
                    
                    # Skip empty rows
                    if not head or not relation or not tail:
                        logger.warning(f"Skipping row {total_rows} with empty values: {row}")
                        errors += 1
                        continue
                    
                    # Insert into staging table
                    try:
                        sqlite_db.execute(
                            "INSERT INTO staging_triples (h_text, r_text, t_text, status) VALUES (?, ?, ?, 'pending')",
                            (head, relation, tail)
                        )
                        successful += 1
                    except sqlite3.IntegrityError:
                        # Handle duplicate triple
                        logger.debug(f"Duplicate triple: {head} {relation} {tail}")
                        duplicates += 1
                    except Exception as e:
                        logger.error(f"Error inserting triple: {str(e)}")
                        errors += 1
                        
                except Exception as e:
                    logger.error(f"Error processing row {total_rows}: {str(e)}")
                    errors += 1
    
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {str(e)}")
        return 0
    
    # Log results
    logger.info(f"CSV ingest summary for {file_path}:")
    logger.info(f"  Total rows processed: {total_rows}")
    logger.info(f"  Successfully staged: {successful}")
    logger.info(f"  Duplicates skipped: {duplicates}")
    logger.info(f"  Errors: {errors}")
    
    return successful

def stage_triples_from_json(file_path):
    """
    Load triples from JSON file into SQLite staging table.
    JSON should be an array of objects with head, relation, tail properties.
    """
    import json
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return 0
    
    # Prepare counters
    total_rows = 0
    successful = 0
    duplicates = 0
    errors = 0
    
    try:
        # Read JSON file
        with open(file_path, 'r', encoding='utf-8') as jsonfile:
            triples = json.load(jsonfile)
            
        if not isinstance(triples, list):
            logger.error("JSON file should contain an array of triples")
            return 0
            
        # Process each triple
        for triple in triples:
            total_rows += 1
            try:
                # Check required fields
                if not all(key in triple for key in ['head', 'relation', 'tail']):
                    logger.warning(f"Skipping triple missing required fields: {triple}")
                    errors += 1
                    continue
                
                # Extract triple
                head = triple['head'].strip()
                relation = triple['relation'].strip()
                tail = triple['tail'].strip()
                
                # Skip empty values
                if not head or not relation or not tail:
                    logger.warning(f"Skipping triple with empty values: {triple}")
                    errors += 1
                    continue
                
                # Insert into staging table
                try:
                    sqlite_db.execute(
                        "INSERT INTO staging_triples (h_text, r_text, t_text, status) VALUES (?, ?, ?, 'pending')",
                        (head, relation, tail)
                    )
                    successful += 1
                except sqlite3.IntegrityError:
                    # Handle duplicate triple
                    logger.debug(f"Duplicate triple: {head} {relation} {tail}")
                    duplicates += 1
                except Exception as e:
                    logger.error(f"Error inserting triple: {str(e)}")
                    errors += 1
                    
            except Exception as e:
                logger.error(f"Error processing triple {total_rows}: {str(e)}")
                errors += 1
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {str(e)}")
        return 0
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {str(e)}")
        return 0
    
    # Log results
    logger.info(f"JSON ingest summary for {file_path}:")
    logger.info(f"  Total triples processed: {total_rows}")
    logger.info(f"  Successfully staged: {successful}")
    logger.info(f"  Duplicates skipped: {duplicates}")
    logger.info(f"  Errors: {errors}")
    
    return successful

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Stage triples for ingestion into the knowledge graph")
    parser.add_argument("--file", required=True, help="Path to CSV or JSON file with triples to ingest")
    
    args = parser.parse_args()
    file_path = args.file
    
    # Check file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        stage_triples_from_csv(file_path)
    elif file_ext == '.json':
        stage_triples_from_json(file_path)
    else:
        logger.error(f"Unsupported file format: {file_ext}. Please use .csv or .json")
        return

if __name__ == "__main__":
    main()