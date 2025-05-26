import os
import argparse
import csv
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# RULE:import-rich-logger-correctly - Use centralized rich logger
from src.app.log import logger, log_and_print
from rich.console import Console

from app.database import sqlite_db

# Initialize rich console for pretty CLI output
console = Console()

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

def create_sample_data():
    """
    Create sample triples for development and testing.
    """
    sample_triples = [
        {"head": "Albert Einstein", "relation": "born_in", "tail": "Germany"},
        {"head": "Albert Einstein", "relation": "developed", "tail": "Theory of Relativity"},
        {"head": "Isaac Newton", "relation": "discovered", "tail": "Law of Gravitation"},
        {"head": "Marie Curie", "relation": "won", "tail": "Nobel Prize"},
        {"head": "Python", "relation": "created_by", "tail": "Guido van Rossum"},
        {"head": "Neo4j", "relation": "is_a", "tail": "Graph Database"},
        {"head": "FAISS", "relation": "developed_by", "tail": "Facebook AI Research"},
        {"head": "Machine Learning", "relation": "subset_of", "tail": "Artificial Intelligence"},
        {"head": "Knowledge Graph", "relation": "stores", "tail": "Structured Information"},
        {"head": "SubgraphRAG", "relation": "uses", "tail": "Knowledge Graph"},
    ]
    
    # Prepare counters
    successful = 0
    duplicates = 0
    errors = 0
    
    logger.info("Creating sample data for development...")
    
    for triple in sample_triples:
        try:
            head = triple['head']
            relation = triple['relation']
            tail = triple['tail']
            
            # Insert into staging table
            try:
                sqlite_db.execute(
                    "INSERT INTO staging_triples (h_text, r_text, t_text, status) VALUES (?, ?, ?, 'pending')",
                    (head, relation, tail)
                )
                successful += 1
                logger.debug(f"Added sample triple: {head} {relation} {tail}")
            except sqlite3.IntegrityError:
                # Handle duplicate triple
                logger.debug(f"Duplicate sample triple: {head} {relation} {tail}")
                duplicates += 1
            except Exception as e:
                logger.error(f"Error inserting sample triple: {str(e)}")
                errors += 1
                
        except Exception as e:
            logger.error(f"Error processing sample triple: {str(e)}")
            errors += 1
    
    # Log results
    logger.info(f"Sample data creation summary:")
    logger.info(f"  Successfully staged: {successful}")
    logger.info(f"  Duplicates skipped: {duplicates}")
    logger.info(f"  Errors: {errors}")
    
    return successful

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Stage triples for ingestion into the knowledge graph")
    parser.add_argument("--file", help="Path to CSV or JSON file with triples to ingest")
    parser.add_argument("--sample", action="store_true", help="Create sample data for development")
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_data()
    elif args.file:
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
    else:
        parser.print_help()
        return

if __name__ == "__main__":
    main()