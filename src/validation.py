from typing import Dict, Any, Optional
from src.app.database import sqlite_db, SQLiteDatabase
from src.app.log import logger

def get_validation_metrics(model_id: str, db: Optional[SQLiteDatabase] = None) -> Dict[str, Any]:
    """
    Get validation metrics for a model.
    
    Args:
        model_id: The ID of the model
        db: Database instance (optional, uses global sqlite_db if not provided)
        
    Returns:
        Dict containing validation metrics
    """
    logger.debug(f"Starting get_validation_metrics for model_id: {model_id}")
    
    # Use provided db instance or fall back to global sqlite_db
    database = db or sqlite_db
    
    if not database:
        logger.error("No database instance available")
        return {
            "status": "error",
            "message": "Database not available",
            "metrics": None
        }
    
    try:
        # Check if model exists in staging_triples (using model_id as a filter)
        model_check = database.fetchone(
            "SELECT COUNT(*) as count FROM staging_triples WHERE h_text LIKE ? OR r_text LIKE ? OR t_text LIKE ?",
            (f"%{model_id}%", f"%{model_id}%", f"%{model_id}%")
        )
        
        if not model_check or model_check[0] == 0:
            logger.warning(f"No data found for model {model_id}")
            return {
                "status": "error",
                "message": f"Model {model_id} not found",
                "metrics": None
            }
            
        # Get validation results from feedback table (using query_id as proxy for model validation)
        validation_results = database.fetchall(
            "SELECT is_correct, comment FROM feedback WHERE query_id LIKE ? ORDER BY timestamp DESC LIMIT 100",
            (f"%{model_id}%",)
        )
        
        if not validation_results:
            logger.info(f"No validation results found for model {model_id}")
            return {
                "status": "error", 
                "message": f"No validation results found for model {model_id}",
                "metrics": None
            }
            
        # Calculate metrics from validation results
        total_validations = len(validation_results)
        correct_validations = sum(1 for result in validation_results if result[0])
        accuracy = correct_validations / total_validations if total_validations > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "total_validations": total_validations,
            "correct_validations": correct_validations,
            "incorrect_validations": total_validations - correct_validations
        }
        
        logger.debug(f"Finished get_validation_metrics for model_id: {model_id}")
        return {
            "status": "success",
            "message": "Validation metrics retrieved successfully",
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting validation metrics: {str(e)}")
        return {
            "status": "error",
            "message": f"Error getting validation metrics: {str(e)}",
            "metrics": None
        } 