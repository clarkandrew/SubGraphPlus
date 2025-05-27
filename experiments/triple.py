#!/usr/bin/env python
"""
triple_cli.py
-------------
End-to-end triple extraction with context-aware NER typing.

• Triples:   Babelscape/rebel-large
• NER tags:  tner/roberta-large-ontonotes5
"""

# ----------------------------------------------------------------------
# STANDARD LIB
# ----------------------------------------------------------------------
import sys
import os
import logging
from typing import List, Dict

# ----------------------------------------------------------------------
# THIRD-PARTY
# ----------------------------------------------------------------------
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    logging as hf_logging
)

from rich.console import Console
from rich.logging import RichHandler

# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------
REBEL_MODEL_NAME: str = "Babelscape/rebel-large"
NER_MODEL_NAME: str = "tner/roberta-large-ontonotes5"
MAX_LEN: int = 256
NUM_BEAMS: int = 3
LOG_LEVEL: int = logging.INFO

# ENV VAR USAGE EXAMPLE
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Rich console
console = Console()

# ----------------------------------------------------------------------
# LOGGER SETUP
# ----------------------------------------------------------------------
def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)]
    )
    hf_logging.set_verbosity_error()
    return logging.getLogger("triple_extractor")

logger = setup_logger()

# ----------------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------------
def load_models():
    try:
        logger.debug(f"Loading REBEL model: {REBEL_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(REBEL_MODEL_NAME)
        rebel_model = AutoModelForSeq2SeqLM.from_pretrained(REBEL_MODEL_NAME)

        logger.debug(f"Loading OntoNotes NER model: {NER_MODEL_NAME}")
        ner_pipe = pipeline("token-classification",
                            model=NER_MODEL_NAME,
                            aggregation_strategy="simple")
        return tokenizer, rebel_model, ner_pipe
    except Exception as e:
        console.print_exception()
        console.print("[bold red]❌ Failed to load models – exiting.")
        sys.exit(1)

# Load once globally
TOKENIZER, REBEL_MODEL, NER_PIPE = load_models()

# ----------------------------------------------------------------------
# REBEL TRIPLET PARSER (from model card)
# ----------------------------------------------------------------------
def extract_triplets(text: str) -> List[Dict[str, str]]:
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(),
                                 'relation': relation.strip(),
                                 'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(),
                                 'relation': relation.strip(),
                                 'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(),
                         'relation': relation.strip(),
                         'tail': object_.strip()})
    return triplets

# ----------------------------------------------------------------------
# CONTEXTUAL NER TYPING
# ----------------------------------------------------------------------
def build_context_ner_map(sentence: str) -> Dict[str, str]:
    ctx_map = {}
    try:
        preds = NER_PIPE(sentence)
        for ent in preds:
            span = ent["word"].strip()
            typ = ent["entity_group"]
            ctx_map[span] = typ
    except Exception:
        logger.exception("Contextual NER failed.")
    return ctx_map

def get_entity_type(span: str) -> str:
    try:
        if not span.strip():
            return "ENTITY"
        pred = NER_PIPE(span)
        return pred[0]["entity_group"] if pred else "ENTITY"
    except Exception:
        logger.exception("Fallback NER failed.")
        return "ENTITY"

# ----------------------------------------------------------------------
# TRIPLE EXTRACTION PIPELINE
# ----------------------------------------------------------------------
def generate_raw(sentence: str) -> str:
    try:
        inputs = TOKENIZER(sentence,
                           max_length=MAX_LEN,
                           padding=True,
                           truncation=True,
                           return_tensors="pt")

        gen_kwargs = dict(max_length=MAX_LEN,
                          num_beams=NUM_BEAMS,
                          length_penalty=0.0)

        generated = REBEL_MODEL.generate(
            inputs["input_ids"].to(REBEL_MODEL.device),
            attention_mask=inputs["attention_mask"].to(REBEL_MODEL.device),
            **gen_kwargs
        )
        decoded = TOKENIZER.batch_decode(generated, skip_special_tokens=False)
        return decoded[0] if decoded else ""
    except Exception:
        logger.exception("Generation failed.")
        raise

def extract(sentence: str, debug: bool = False) -> List[Dict[str, str]]:
    try:
        ctx_types = build_context_ner_map(sentence)
        raw = generate_raw(sentence)
        triples = extract_triplets(raw)

        for t in triples:
            head = t["head"]
            tail = t["tail"]
            t["head_type"] = ctx_types.get(head, get_entity_type(head))
            t["tail_type"] = ctx_types.get(tail, get_entity_type(tail))

        if debug:
            console.print("\n[dim]RAW OUTPUT:[/]\n", raw)

        return triples
    except Exception:
        logger.exception("Full extraction failed.")
        raise

# ----------------------------------------------------------------------
# CLI DISPLAY
# ----------------------------------------------------------------------
def display_triples(triples: List[Dict[str, str]]):
    if not triples:
        console.print("[bold yellow]⚠ No triples found.")
        return

    console.print("\n[bold cyan]EXTRACTED TRIPLES[/]")
    console.print("[dim]" + "-"*90)

    for i, t in enumerate(triples, 1):
        subj_type = t.get("head_type", "ENTITY")
        obj_type = t.get("tail_type", "ENTITY")

        console.print(
            f"[green]{i}.[/] "
            f"[bold]Subject:[/] [dim italic]{subj_type}[/] [cyan]{t['head']}[/]  "
            f"[bold]Relation:[/] [blue]{t['relation']}[/]  "
            f"[bold]Object:[/] [dim italic]{obj_type}[/] [magenta]{t['tail']}[/]"
        )

# ----------------------------------------------------------------------
# QUESTIONARY INPUT
# ----------------------------------------------------------------------
def prompt_sentence() -> str:
    text = 'Punta Cana is a resort town in the municipality of Higüey, in La Altagracia Province, the easternmost province of the Dominican Republic.'

    return text

# ----------------------------------------------------------------------
# MAIN ENTRY
# ----------------------------------------------------------------------
def main():
    try:
        # Sentence input
        sentence = " ".join(arg for arg in sys.argv[1:] if not arg.startswith("--")).strip()
        debug = "--debug" in sys.argv

        if not sentence:
            sentence = prompt_sentence()
            if not sentence:
                console.print("[bold red]No input provided. Exiting.")
                return

        console.print(f"[bold green]Input:[/] {sentence}")
        triples = extract(sentence, debug=debug)
        display_triples(triples)

    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user.")
    except Exception:
        console.print_exception()
        console.print("[bold red]Unexpected error occurred.")

# ----------------------------------------------------------------------
# SCRIPT ENTRY
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
