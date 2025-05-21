import argparse
import json
import time
import logging
import csv
import os
import sys
import statistics
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add parent directory to path so we can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.models import QueryRequest
from app.database import neo4j_db, sqlite_db
from app.retriever import hybrid_retrieve_v2
from app.utils import extract_query_entities, link_entities_v2, triples_to_graph_data
from app.ml.llm import generate_answer
from app.verify import validate_llm_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'benchmark.log'))
    ]
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Benchmark evaluation for SubgraphRAG+"""

    def __init__(self, input_file: str, output_file: str, metrics_file: str):
        """
        Initialize benchmark runner
        
        Args:
            input_file: Path to input file with test questions
            output_file: Path to output file for results
            metrics_file: Path to metrics output file
        """
        self.input_file = input_file
        self.output_file = output_file
        self.metrics_file = metrics_file
        self.results = []
        self.metrics = {
            "total_questions": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "retrieval_empty_count": 0,
            "entity_linking_failure_count": 0,
            "llm_failure_count": 0,
            "other_error_count": 0,
            "total_duration": 0,
            "avg_duration": 0,
            "avg_retrieved_triples": 0,
            "avg_cited_triples": 0,
            "p50_latency": 0,
            "p90_latency": 0,
            "p95_latency": 0,
            "p99_latency": 0,
        }
        self.latencies = []
        self.retrieved_triples_counts = []
        self.cited_triples_counts = []

    def load_test_questions(self) -> List[Dict[str, Any]]:
        """Load test questions from input file"""
        questions = []
        
        # Determine file format from extension
        if self.input_file.endswith('.json'):
            with open(self.input_file, 'r') as f:
                questions = json.load(f)
        elif self.input_file.endswith('.csv'):
            with open(self.input_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    questions.append(row)
        else:
            raise ValueError(f"Unsupported file format: {self.input_file}")
        
        return questions

    def run_single_query(self, question: str) -> Dict[str, Any]:
        """
        Run a single query through the pipeline
        
        Args:
            question: Question text
            
        Returns:
            Dictionary with query results and metrics
        """
        start_time = time.time()
        result = {
            "question": question,
            "success": False,
            "answers": [],
            "citations": [],
            "retrieved_triple_count": 0,
            "cited_triple_count": 0,
            "error": None,
            "duration_seconds": 0
        }
        
        try:
            # Step 1: Extract entities from question
            potential_entities = extract_query_entities(question)
            
            # Step 2: Link entities to knowledge graph
            linked_entities = []
            for entity_text in potential_entities:
                entity_links = link_entities_v2(entity_text, question)
                linked_entities.extend([entity_id for entity_id, conf in entity_links if conf >= 0.75])
            
            if not linked_entities:
                result["error"] = "NO_ENTITY_MATCH"
                return result
            
            # Step 3: Retrieve relevant triples
            retrieved_triples = hybrid_retrieve_v2(question, linked_entities)
            result["retrieved_triple_count"] = len(retrieved_triples)
            
            # Step 4: Generate answer with LLM
            system_message = "You are a precise, factual question-answering assistant. Your knowledge is strictly limited to the triples provided below. Respond concisely with factual information only."
            
            # Format prompt
            prompt = f"{system_message}\n\nBased *only* on the following information, answer the question.\n"
            prompt += "Cite the ID of each triple you use in your reasoning using the format (id=XXX).\n"
            prompt += "Available Triples:\n"
            
            for triple in retrieved_triples:
                prompt += f"(id={triple.id}) {triple.head_name} {triple.relation_name} {triple.tail_name}.\n"
            
            prompt += f"\nQuestion: {question}\n\n"
            prompt += "Answer directly. If the information is not present in the triples, state \"Information not available in provided context.\"\n"
            prompt += "Format your final answer(s) on new lines, each starting with \"ans: \". Example:\n"
            prompt += "ans: Elon Musk (id=123)\n"
            prompt += "ans: SpaceX (id=456)"
            
            # Generate answer
            answer_text = generate_answer(prompt)
            
            # Step 5: Validate answer
            triple_ids = {t.id for t in retrieved_triples}
            answers, cited_ids, trust_level = validate_llm_output(answer_text, triple_ids)
            
            # Store results
            result["success"] = True
            result["answers"] = answers
            result["citations"] = cited_ids
            result["cited_triple_count"] = len(cited_ids)
            result["trust_level"] = trust_level
            
        except Exception as e:
            # Track specific error types
            error_type = type(e).__name__
            result["error"] = f"{error_type}: {str(e)}"
            
        finally:
            # Calculate duration
            end_time = time.time()
            duration = end_time - start_time
            result["duration_seconds"] = duration
        
        return result

    def run_benchmark(self):
        """Run benchmark on all test questions"""
        logger.info(f"Starting benchmark with input file: {self.input_file}")
        
        # Load test questions
        questions = self.load_test_questions()
        total_questions = len(questions)
        self.metrics["total_questions"] = total_questions
        
        logger.info(f"Loaded {total_questions} test questions")
        
        # Run benchmark for each question
        for i, question_data in enumerate(questions):
            # Extract question text
            if isinstance(question_data, dict) and "question" in question_data:
                question = question_data["question"]
                question_id = question_data.get("id", f"q{i+1}")
            else:
                # If question_data is a string
                question = question_data
                question_id = f"q{i+1}"
            
            logger.info(f"Processing question {i+1}/{total_questions}: {question}")
            
            # Run the query
            result = self.run_single_query(question)
            
            # Add question ID and timestamp
            result["id"] = question_id
            result["timestamp"] = datetime.now().isoformat()
            
            # Store result
            self.results.append(result)
            
            # Update metrics
            if result["success"]:
                self.metrics["successful_queries"] += 1
                self.retrieved_triples_counts.append(result["retrieved_triple_count"])
                self.cited_triples_counts.append(result["cited_triple_count"])
            else:
                self.metrics["failed_queries"] += 1
                
                # Track error types
                error = result.get("error", "")
                if "RetrievalEmpty" in error:
                    self.metrics["retrieval_empty_count"] += 1
                elif "EntityLinkingError" in error or "NO_ENTITY_MATCH" in error:
                    self.metrics["entity_linking_failure_count"] += 1
                elif "LLM" in error:
                    self.metrics["llm_failure_count"] += 1
                else:
                    self.metrics["other_error_count"] += 1
            
            # Track latency
            self.latencies.append(result["duration_seconds"])
            self.metrics["total_duration"] += result["duration_seconds"]
            
            # Log progress
            logger.info(f"Completed question {i+1}/{total_questions} in {result['duration_seconds']:.2f}s")
            
        # Calculate aggregate metrics
        if self.metrics["successful_queries"] > 0:
            self.metrics["avg_retrieved_triples"] = sum(self.retrieved_triples_counts) / self.metrics["successful_queries"]
            self.metrics["avg_cited_triples"] = sum(self.cited_triples_counts) / self.metrics["successful_queries"]
        
        if self.latencies:
            self.metrics["avg_duration"] = self.metrics["total_duration"] / total_questions
            self.latencies.sort()
            self.metrics["p50_latency"] = self.latencies[int(total_questions * 0.50)]
            self.metrics["p90_latency"] = self.latencies[int(total_questions * 0.90)]
            self.metrics["p95_latency"] = self.latencies[int(total_questions * 0.95)]
            self.metrics["p99_latency"] = self.latencies[int(total_questions * 0.99)]
        
        # Save results
        self.save_results()
        self.save_metrics()
        
        logger.info(f"Benchmark complete. Results saved to {self.output_file}")
        logger.info(f"Metrics saved to {self.metrics_file}")

    def save_results(self):
        """Save benchmark results to output file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def save_metrics(self):
        """Save benchmark metrics to metrics file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)


def main():
    """Main entry point for benchmark script"""
    parser = argparse.ArgumentParser(description="SubgraphRAG+ Benchmark Evaluation")
    parser.add_argument("--input", required=True, help="Input file with test questions (JSON or CSV)")
    parser.add_argument("--output", default="evaluation/results.json", help="Output file for results (JSON)")
    parser.add_argument("--metrics", default="evaluation/metrics.json", help="Output file for metrics (JSON)")
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    
    # Run benchmark
    benchmark = BenchmarkRunner(args.input, args.output, args.metrics)
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()