import re
from typing import Dict, List, Set, Tuple, Optional, Any

# RULE:import-rich-logger-correctly - Use centralized rich logger
from .log import logger, log_and_print
from rich.console import Console

# Initialize rich console for pretty CLI output
console = Console()

# Regex for valid answer line
ANSWER_LINE_REGEX = re.compile(r"^ans:\s*(.+?)\s*\(id=(\d+(?:,\s*\d+)*)\)$")

def validate_llm_output(llm_response_text: str, provided_triple_ids: Set[str]) -> Tuple[List[str], List[str], str]:
    """
    Validate and parse LLM output to ensure it follows the expected format
    and only cites provided triple IDs.

    Args:
        llm_response_text: Raw text response from LLM
        provided_triple_ids: Set of triple IDs provided in the context

    Returns:
        Tuple of (answers, cited_ids_list, trust_level)
    """
    answers = []
    cited_ids_flat = []
    trust = "high"

    # Basic Injection Guard
    if "\nTriples:" in llm_response_text.split("Question:")[0]:  # Check if LLM tries to inject its own triples section
        logger.warning("Possible injection detected in LLM response")
        return [], [], "low_injection_detected"

    # Sanitize common Markdown/comment patterns that might be used for injection
    sanitized_response = llm_response_text.replace("#", "").replace("/*", "").replace("*/", "")

    # Check for "Information not available" response
    if "Information not available" in sanitized_response:
        logger.info("LLM indicates information not available in context")
        return ["Information not available in provided context."], [], "high"

    # Process response line by line
    for line in sanitized_response.splitlines():
        if line.strip().startswith("ans:"):
            match = ANSWER_LINE_REGEX.match(line)
            if not match:
                logger.warning(f"Malformed answer line: {line}")
                trust = "low_malformed_ans_line"  # Entire response suspect
                continue

            answer_text = match.group(1).strip()
            cited_ids_str = match.group(2).split(',')
            current_line_cited_ids = []

            for c_id_str in cited_ids_str:
                c_id = c_id_str.strip()
                if c_id not in provided_triple_ids:
                    logger.warning(f"Invalid citation ID: {c_id}")
                    trust = "low_invalid_citation"  # Specific citation is bad

                current_line_cited_ids.append(c_id)

            answers.append(answer_text)
            cited_ids_flat.extend(current_line_cited_ids)

    if not answers and "Information not available" not in sanitized_response:
        logger.warning("No structured answers found in response")
        trust = "low_no_structured_answer"

    return answers, list(set(cited_ids_flat)), trust


def format_prompt(system_message: str, triples: List[Dict[str, Any]], user_question: str) -> str:
    """
    Format prompt for the LLM

    Args:
        system_message: System message to guide the LLM behavior
        triples: List of triples to include in context
        user_question: User's question

    Returns:
        Formatted prompt string
    """
    prompt = f"{system_message}\n\n"
    prompt += "Based *only* on the following information, answer the question.\n"
    prompt += "Cite the ID of each triple you use in your reasoning using the format (id=XXX).\n"

    prompt += "Available Triples:\n"
    for triple in triples:
        prompt += f"(id={triple['id']}) {triple['head_name']} {triple['relation_name']} {triple['tail_name']}.\n"

    prompt += f"\nQuestion: {user_question}\n\n"
    prompt += "Answer directly. If the information is not present in the triples, state \"Information not available in provided context.\"\n"
    prompt += "Format your final answer(s) on new lines, each starting with \"ans: \". Example:\n"
    prompt += "ans: Elon Musk (id=123)\n"
    prompt += "ans: SpaceX (id=456)"

    return prompt
