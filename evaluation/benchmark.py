import argparse
import json
import time
import csv
import os
import sys
import statistics
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Set, Optional

# Add parent directory to path so we can import app modules
sys.path.append(str(Path(__file__).parent.parent))

# RULE:import-rich-logger-correctly - Use centralized rich logger
from src.app.log import logger, log_and_print
from rich.console import Console

from app.models import QueryRequest
from app.database import neo4j_db, sqlite_db
from app.retriever import hybrid_retrieve_v2
from app.utils import extract_query_entities, link_entities_v2, triples_to_graph_data
from app.ml.llm import generate_answer
from app.verify import validate_llm_output

# Initialize rich console for pretty CLI output
console = Console()


class BenchmarkRunner:
    """Benchmark evaluation for SubgraphRAG+"""

    def __init__(self, input_file: str, output_file: str, metrics_file: str, ground_truth_file: str = None):
        """
        Initialize benchmark runner
        
        Args:
            input_file: Path to input file with test questions
            output_file: Path to output file for results
            metrics_file: Path to metrics output file
            ground_truth_file: Optional path to ground truth file for advanced metrics
        """
        self.input_file = input_file
        self.output_file = output_file
        self.metrics_file = metrics_file
        self.ground_truth_file = ground_truth_file
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
            # Advanced metrics
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "entity_linking_accuracy": 0.0,
            "answer_exactness": 0.0,
            "robustness_score": 0.0,
            # Adversarial metrics
            "adversarial_success_rate": 0.0,
            "hallucination_rate": 0.0,
        }
        self.latencies = []
        self.retrieved_triples_counts = []
        self.cited_triples_counts = []
        self.ground_truth = self.load_ground_truth() if ground_truth_file else {}

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
        
    def load_ground_truth(self) -> Dict[str, Any]:
        """Load ground truth data if available"""
        if not self.ground_truth_file:
            return {}
            
        # Load ground truth from file
        try:
            if self.ground_truth_file.endswith('.json'):
                with open(self.ground_truth_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Unsupported ground truth file format: {self.ground_truth_file}")
                return {}
        except Exception as e:
            logger.error(f"Error loading ground truth file: {str(e)}")
            return {}

    def run_single_query(self, question: str, question_id: str = None, is_adversarial: bool = False) -> Dict[str, Any]:
        """
        Run a single query through the pipeline
        
        Args:
            question: Question text
            question_id: Optional question ID for ground truth lookup
            is_adversarial: Whether this is an adversarial test question
            
        Returns:
            Dictionary with query results and metrics
        """
        start_time = time.time()
        result = {
            "question": question,
            "question_id": question_id,
            "success": False,
            "answers": [],
            "citations": [],
            "retrieved_triple_count": 0,
            "cited_triple_count": 0,
            "error": None,
            "duration_seconds": 0,
            "is_adversarial": is_adversarial,
            "expected_entities": [],
            "linked_entities": [],
            "entity_linking_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
        
        try:
            # Step 1: Extract entities from question
            potential_entities = extract_query_entities(question)
            
            # Step 2: Link entities to knowledge graph
            linked_entities = []
            linked_entities_with_conf = []
            for entity_text in potential_entities:
                entity_links = link_entities_v2(entity_text, question)
                linked_entities.extend([entity_id for entity_id, conf in entity_links if conf >= 0.75])
                linked_entities_with_conf.extend(entity_links)
            
            # Store all linked entities with confidence for metrics
            result["linked_entities"] = [(entity_id, conf) for entity_id, conf in linked_entities_with_conf]
            
            # Check for ground truth expected entities if available
            if question_id and question_id in self.ground_truth:
                expected_entities = self.ground_truth[question_id].get("expected_entities", [])
                result["expected_entities"] = expected_entities
                
                # Calculate entity linking accuracy if expected entities exist
                if expected_entities:
                    # Simple overlap metric for now
                    found = set(linked_entities)
                    expected = set(expected_entities)
                    if expected:
                        result["entity_linking_accuracy"] = len(found.intersection(expected)) / len(expected)
            
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
            
            # Calculate precision, recall, F1 if ground truth available
            if question_id and question_id in self.ground_truth:
                # Get expected triple citations from ground truth
                expected_citations = set(self.ground_truth[question_id].get("expected_triple_ids", []))
                expected_answers = set(self.ground_truth[question_id].get("expected_answers", []))
            
                # Calculate metrics if expected citations or answers are provided
                if expected_citations:
                    result["precision"], result["recall"], result["f1_score"] = self.calculate_citation_metrics(
                        set(cited_ids) if cited_ids else set(), expected_citations
                    )
            
                # Check for hallucinations - answers not backed by expected citations
                if expected_answers:
                    result["answer_exactness"] = self.calculate_answer_exactness(answers, expected_answers)
                    result["hallucinated"] = len(set(answers if answers else []) - expected_answers) > 0
            
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

    def calculate_citation_metrics(self, predicted_citations: Set[str], expected_citations: Set[str]) -> Tuple[float, float, float]:
        """Calculate precision, recall, F1 for citation prediction"""
        if not predicted_citations and not expected_citations:
            return 1.0, 1.0, 1.0  # Perfect score for empty sets
            
        if not predicted_citations:
            return 0.0, 0.0, 0.0  # No predictions made
            
        if not expected_citations:
            return 0.0, 1.0, 0.0  # Predicted when nothing expected
        
        # True positives: citations that are both predicted and expected
        true_positives = len(predicted_citations.intersection(expected_citations))
        
        # Calculate metrics
        precision = true_positives / len(predicted_citations) if predicted_citations else 0
        recall = true_positives / len(expected_citations) if expected_citations else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1_score
        
    def calculate_answer_exactness(self, predicted_answers: List[str], expected_answers: Set[str]) -> float:
        """Calculate how exact the answer matches expected answers"""
        if not predicted_answers:
            return 0.0
            
        if not expected_answers:
            return 0.0
            
        # Simple match percentage for now
        # For better results, consider semantic similarity or fuzzy matching
        matches = sum(1 for answer in predicted_answers if answer in expected_answers)
        return matches / max(len(predicted_answers), len(expected_answers)) if max(len(predicted_answers), len(expected_answers)) > 0 else 0.0
    
    def run_benchmark(self):
        """Run benchmark on all test questions"""
        logger.info(f"Starting benchmark with input file: {self.input_file}")
        
        # Load test questions
        questions = self.load_test_questions()
        total_questions = len(questions)
        self.metrics["total_questions"] = total_questions
        
        logger.info(f"Loaded {total_questions} test questions")
        
        # Track metrics for advanced calculations
        adversarial_count = 0
        adversarial_success = 0
        hallucination_count = 0
        precision_values = []
        recall_values = []
        f1_values = []
        entity_linking_accuracy_values = []
        
        # Run benchmark for each question
        for i, question_data in enumerate(questions):
            # Extract question text
            if isinstance(question_data, dict) and "question" in question_data:
                question = question_data["question"]
                question_id = question_data.get("id", f"q{i+1}")
                is_adversarial = question_data.get("is_adversarial", False)
            else:
                # If question_data is a string
                question = question_data
                question_id = f"q{i+1}"
                is_adversarial = False
            
            logger.info(f"Processing question {i+1}/{total_questions}: {question}")
            
            # Run the query
            result = self.run_single_query(question, question_id, is_adversarial)
            
            # Add question ID and timestamp
            result["id"] = question_id
            result["timestamp"] = datetime.now().isoformat()
            
            # Store result
            self.results.append(result)
            
            # Track advanced metrics
            if is_adversarial:
                adversarial_count += 1
                if result["success"]:
                    adversarial_success += 1
                    
            if result.get("hallucinated", False):
                hallucination_count += 1
                
            if result["precision"] > 0:
                precision_values.append(result["precision"])
                
            if result["recall"] > 0:
                recall_values.append(result["recall"])
                
            if result["f1_score"] > 0:
                f1_values.append(result["f1_score"])
                
            if result["entity_linking_accuracy"] > 0:
                entity_linking_accuracy_values.append(result["entity_linking_accuracy"])
            
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
            
        # Calculate advanced metrics
        if precision_values:
            self.metrics["precision"] = sum(precision_values) / len(precision_values)
        else:
            self.metrics["precision"] = 0.0
        
        if recall_values:
            self.metrics["recall"] = sum(recall_values) / len(recall_values)
        else:
            self.metrics["recall"] = 0.0
        
        if f1_values:
            self.metrics["f1_score"] = sum(f1_values) / len(f1_values)
        else:
            self.metrics["f1_score"] = 0.0
        
        if entity_linking_accuracy_values:
            self.metrics["entity_linking_accuracy"] = sum(entity_linking_accuracy_values) / len(entity_linking_accuracy_values)
        else:
            self.metrics["entity_linking_accuracy"] = 0.0
        
        if adversarial_count > 0:
            self.metrics["adversarial_success_rate"] = adversarial_success / adversarial_count
        
        if total_questions > 0:
            self.metrics["hallucination_rate"] = hallucination_count / total_questions
        
        # Calculate combined robustness score (weighted average of precision, recall, and adversarial success)
        if self.metrics["precision"] > 0 or self.metrics["recall"] > 0:
            self.metrics["robustness_score"] = (
                (0.4 * self.metrics["precision"]) + 
                (0.4 * self.metrics["recall"]) + 
                (0.2 * (1 - self.metrics.get("hallucination_rate", 0)))
            )
        
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
    parser.add_argument("--ground-truth", help="Optional ground truth file for advanced metrics")
    parser.add_argument("--detailed-report", action="store_true", help="Generate a detailed HTML report")
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    
    # Run benchmark
    benchmark = BenchmarkRunner(args.input, args.output, args.metrics, args.ground_truth)
    benchmark.run_benchmark()
    
    # Generate detailed HTML report if requested
    if args.detailed_report:
        try:
            # Check for required packages
            missing_packages = []
            try:
                from jinja2 import Template
            except ImportError:
                missing_packages.append("jinja2")
                
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                missing_packages.append("matplotlib")
                
            try:
                import base64
                from io import BytesIO
            except ImportError:
                missing_packages.append("base64/io")
                
            if missing_packages:
                logger.warning(f"Cannot generate HTML report. Missing packages: {', '.join(missing_packages)}")
                logger.warning("Install with: pip install jinja2 matplotlib")
                sys.exit(1)
            
            # Generate charts
            def generate_chart(data, title, filename):
                plt.figure(figsize=(10, 6))
                
                # Basic charts based on data type
                if isinstance(data, list) and data:
                    plt.plot(data)
                elif isinstance(data, dict) and data:
                    plt.bar(list(data.keys())[:20], list(data.values())[:20])  # Limit to 20 items max
                else:
                    # Create empty chart with message
                    plt.text(0.5, 0.5, "No data available", horizontalalignment='center', verticalalignment='center')
                
                plt.title(title)
                plt.tight_layout()
                
                # Save to BytesIO
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Convert to base64 for embedding
                img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()
                
                return f"data:image/png;base64,{img_base64}"
            
            # Create charts
            charts = {}
            try:
                charts["latency"] = generate_chart(benchmark.latencies, "Query Latency Distribution", "latency.png")
            except Exception as e:
                logger.warning(f"Failed to generate latency chart: {str(e)}")
                
            try:
                charts["retrieval_count"] = generate_chart(benchmark.retrieved_triples_counts, "Retrieved Triples Count", "retrieval.png")
            except Exception as e:
                logger.warning(f"Failed to generate retrieval count chart: {str(e)}")
                
            try:
                charts["cited_count"] = generate_chart(benchmark.cited_triples_counts, "Cited Triples Count", "cited.png")
            except Exception as e:
                logger.warning(f"Failed to generate cited count chart: {str(e)}")
            
            # Load template
            template_path = os.path.join(os.path.dirname(__file__), "report_template.html")
            if not os.path.exists(template_path):
                # Create basic template
                template_str = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>SubgraphRAG+ Benchmark Results</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        .metric { margin-bottom: 10px; }
                        .chart { margin: 20px 0; max-width: 100%; }
                        table { border-collapse: collapse; width: 100%; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        tr:nth-child(even) { background-color: #f2f2f2; }
                        th { background-color: #4CAF50; color: white; }
                        .success { color: green; }
                        .failure { color: red; }
                    </style>
                </head>
                <body>
                    <h1>SubgraphRAG+ Benchmark Results</h1>
                    <h2>Summary Metrics</h2>
                    <div class="metrics">
                        {% for key, value in metrics.items() %}
                            <div class="metric"><strong>{{ key }}:</strong> 
                                {% if value is number %}
                                    {{ "%.4f"|format(value) if value < 1 else "%.2f"|format(value) }}
                                {% else %}
                                    {{ value }}
                                {% endif %}
                            </div>
                        {% endfor %}
                    </div>
                    
                    <h2>Charts</h2>
                    {% for name, img_data in charts.items() %}
                        <h3>{{ name }}</h3>
                        <div class="chart">
                            <img src="{{ img_data }}" alt="{{ name }} chart">
                        </div>
                    {% endfor %}
                    
                    <h2>Results</h2>
                    <table>
                        <tr>
                            <th>Question</th>
                            <th>Success</th>
                            <th>Answer</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1</th>
                            <th>Latency (s)</th>
                            <th>Error</th>
                        </tr>
                        {% for result in results %}
                            <tr>
                                <td>{{ result.question }}</td>
                                <td class="{{ 'success' if result.success else 'failure' }}">
                                    {{ "✓" if result.success else "✗" }}
                                </td>
                                <td>{{ ", ".join(result.answers) if result.answers else "-" }}</td>
                                <td>{{ "%.4f"|format(result.precision) }}</td>
                                <td>{{ "%.4f"|format(result.recall) }}</td>
                                <td>{{ "%.4f"|format(result.f1_score) }}</td>
                                <td>{{ "%.2f"|format(result.duration_seconds) }}</td>
                                <td>{{ result.error or "-" }}</td>
                            </tr>
                        {% endfor %}
                    </table>
                </body>
                </html>
                """
            else:
                with open(template_path, 'r') as f:
                    template_str = f.read()
            
            # Render template
            template = Template(template_str)
            html = template.render(
                metrics=benchmark.metrics,
                results=benchmark.results,
                charts=charts
            )
            
            # Write HTML report
            report_path = os.path.join(os.path.dirname(args.output), "benchmark_report.html")
            with open(report_path, 'w') as f:
                f.write(html)
                
            logger.info(f"Detailed report generated at {report_path}")
        except ImportError:
            logger.warning("Could not generate HTML report. Required packages: jinja2, matplotlib")


if __name__ == "__main__":
    main()