import re, sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging

logging.set_verbosity_error()   # silence warnings

MODEL_ID = "Babelscape/rebel-large"

# ----- load once -------------------------------------------------------------
print("Loading REBEL-large … this can take 30‒40 s the first time.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

def extract_triplets(text, debug=False):
    triplets = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'

    if debug:
        print(f"Debug: Processing text: {text}")

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if debug:
            print(f"Debug: Token='{token}', Current='{current}', Subject='{subject}', Object='{object_}', Relation='{relation}'")

        if token == "<triplet>":
            current = 't'  # Start with subject after <triplet>
            if relation != '' and subject != '' and object_ != '':
                triplets.append({'head': subject.strip(), 'relation': relation.strip(), 'tail': object_.strip()})
                if debug:
                    print(f"Debug: Added triplet - head:{subject.strip()}, relation:{relation.strip()}, tail:{object_.strip()}")
            relation = ''
            subject = ''
            object_ = ''
        elif token == "<subj>":
            current = 's'  # Switch to object after <subj>
        elif token == "<obj>":
            current = 'o'  # Switch to relation after <obj>
        else:
            if current == 't':
                subject += ' ' + token if subject else token
            elif current == 's':
                object_ += ' ' + token if object_ else token
            elif current == 'o':
                relation += ' ' + token if relation else token

    # Add final triplet if exists
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'relation': relation.strip(), 'tail': object_.strip()})
        if debug:
            print(f"Debug: Final triplet - head:{subject.strip()}, relation:{relation.strip()}, tail:{object_.strip()}")

    return triplets

# Add entity typing function
def get_entity_type(entity_name: str) -> str:
    """
    Simple entity typing - in production this uses the schema-driven approach
    """
    # Load entity types from config (simplified for demo)
    entity_types = {
        "Moses": "Person",
        "Aaron": "Person", 
        "Abraham": "Person",
        "Israelites": "Organization",
        "Hebrews": "Organization",
        "Egypt": "Location",
        "Red Sea": "Location",
        "Jerusalem": "Location",
        "Exodus": "Event",
        "Passover": "Event",
        "Ten Commandments": "Concept",
        "Covenant": "Concept"
    }
    
    return entity_types.get(entity_name, "Entity")

def extract_triples(text: str, debug=False):
    # Tokenize input text
    model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors='pt')

    # Generation parameters
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 1,
    }

    # Generate tokens
    generated_tokens = model.generate(
        model_inputs["input_ids"].to(model.device),
        attention_mask=model_inputs["attention_mask"].to(model.device),
        **gen_kwargs,
    )

    # Decode with special tokens
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # Extract triplets from first prediction
    raw_output = decoded_preds[0] if decoded_preds else ""
    triplets = extract_triplets(raw_output, debug=debug)
    
    # Add entity types to each triple
    for triple in triplets:
        triple['head_type'] = get_entity_type(triple['head'])
        triple['tail_type'] = get_entity_type(triple['tail'])

    return triplets, raw_output

if __name__ == "__main__":
    # Check for debug flag
    debug_mode = "--debug" in sys.argv
    if debug_mode:
        sys.argv.remove("--debug")

    sentence = " ".join(sys.argv[1:]) or \
        "Moses parted the Red Sea and led the Israelites across."

    print(f"\nInput: {sentence}")
    if debug_mode:
        print("DEBUG MODE: Enabled")

    triples, raw = extract_triples(sentence, debug=debug_mode)

    print("\nRAW MODEL OUTPUT")
    print("----------------")
    print(raw)

    print("\nEXTRACTED TRIPLES")
    print("-----------------")
    if triples:
        for i, triple in enumerate(triples, 1):
            head = triple['head']
            rel = triple['relation']
            tail = triple['tail']
            head_type = triple.get('head_type', 'Entity')
            tail_type = triple.get('tail_type', 'Entity')
            
            print(f"{i}. Subject: {head:<20} Relation: {rel:<15} Object: {tail}")
            print(f"   Types:   {head_type:<20}           {tail_type}")
            print()

            # Check for potentially malformed triplets and suggest corrections
            if debug_mode and (len(head.split()) > 3 or len(tail.split()) > 3):
                print(f"   ⚠️  This triplet may be malformed - very long subject/object")
    else:
        print("⚠️  No valid triplets found.")
