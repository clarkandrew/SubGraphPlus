# SubgraphRAG+ Evaluation Guide

## Overview

This guide explains how to run the evaluation/benchmarking framework for SubgraphRAG+.

---

## Prerequisites

- The API and all dependencies (Neo4j, FAISS, etc.) must be running.
- The knowledge graph should be populated (e.g., via `make ingest-sample`).
- The evaluation script and sample questions are in the `evaluation/` directory.

---

## Running the Evaluation

### **Step 1: Prepare the System**

- Ensure the API server is running (`make serve` or via Docker).
- Ensure Neo4j and FAISS are up-to-date and consistent (run reconciliation if needed).

### **Step 2: Run the Benchmark Script**

```bash
python evaluation/benchmark.py --input evaluation/sample_questions.json
```

- By default, results will be saved to `evaluation/results.json` and metrics to `evaluation/metrics.json`.
- You can specify custom output paths with `--output` and `--metrics`.

### **Step 3: Review Results**

- Open `evaluation/results.json` for per-question outputs (answers, citations, errors, latency).
- Open `evaluation/metrics.json` for aggregate statistics (accuracy, latency, retrieval stats).

---

## Customizing Evaluation

- You can provide your own question set in JSON or CSV format.
- Each entry should have at least a `question` field (optionally `id`, `expected_entities`, etc.).

---

## Example Output

- **results.json**: List of dicts, one per question, with answers, citations, timings, and errors.
- **metrics.json**: Aggregate statistics (success rate, average latency, p95 latency, etc.).

---

## Troubleshooting

- If you see errors about missing entities or retrieval failures, check that your KG is populated.
- For LLM errors, ensure your backend (OpenAI, HF, MLX) is configured and available.
- Check logs in `logs/` for more details.

---

## Advanced

- You can run the evaluation inside Docker by attaching to the running container and executing the script.
- For large-scale or automated evals, integrate this script into your CI/CD pipeline.
