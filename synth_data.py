import random
import spacy

# Make sure you've already done:
nlp = spacy.load("en_core_web_sm")

FANBOYS = {"and", "but", "or", "so", "yet", "for", "nor"}

def has_finite_verb(span):
    """Heuristic: does this span contain a verb or auxiliary?"""
    return any(tok.pos_ in {"VERB", "AUX"} for tok in span)


def make_comma_splice(sentence):
    """
    Convert 'I went home and I slept.' → 'I went home, I slept.'
    Only valid if both sides of the conjunction are clauses
    with finite verbs and the sentence ends in punctuation.
    """
    doc = nlp(sentence)

    # Must end in punctuation
    if not doc or doc[-1].pos_ != "PUNCT":
        return None

    for tok in doc:
        # Find coordinating conjunction from FANBOYS list
        if tok.pos_ == "CCONJ" and tok.text.lower() in FANBOYS:
            left = doc[:tok.i]
            right = doc[tok.i + 1:-1]  # exclude final punctuation

            if not left or not right:
                continue
            if not (has_finite_verb(left) and has_finite_verb(right)):
                continue

            left_text = " ".join(t.text for t in left).strip()
            right_text = " ".join(t.text for t in right).strip()
            end_punct = doc[-1].text

            return f"{left_text}, {right_text}{end_punct}"

    return None


def remove_random_comma(sentence):
    """
    Remove one random comma from the sentence.
    """
    doc = nlp(sentence)
    comma_indices = [i for i, t in enumerate(doc) if t.text == ","]

    if not comma_indices:
        return None

    idx = random.choice(comma_indices)
    tokens = [t.text for t in doc]
    del tokens[idx]

    # Basic cleanup
    corrupted = " ".join(tokens)
    corrupted = (corrupted.replace(" ,", ",")
                           .replace(" .", ".")
                           .replace(" !", "!")
                           .replace(" ?", "?"))
    return corrupted


def insert_random_comma(sentence):
    """
    Insert a comma in a random valid place, but only if the
    sentence contains NO commas already.
    """
    doc = nlp(sentence)
    if any(t.text == "," for t in doc):
        return None

    # Insert comma between non-punctuation tokens (not edges)
    positions = [
        i for i in range(1, len(doc) - 1)
        if not doc[i - 1].is_punct and not doc[i].is_punct
    ]

    if not positions:
        return None

    idx = random.choice(positions)

    tokens = [t.text for t in doc]
    tokens = tokens[:idx] + [","] + tokens[idx:]

    corrupted = " ".join(tokens)
    corrupted = (corrupted.replace(" ,", ",")
                           .replace(" .", ".")
                           .replace(" !", "!")
                           .replace(" ?", "?"))

    return corrupted


import pandas as pd

def build_synthetic_comma_dataset(clean_sentences, max_per_type=5000):
    """
    Build a DataFrame of synthetic comma errors.

    clean_sentences : list/iterable of clean English sentences
    max_per_type    : number of synthetic examples per error type

    Returns DataFrame with:
        - sentence : the original or corrupted sentence
        - label    : 0 = original (correct), 1 = corrupted (error)
        - error_type : 'orig', 'comma_splice', 'comma_deleted', 'comma_inserted'
        - synthetic_source : category used for generation
    """
    rows = []
    count_splice = 0
    count_del = 0
    count_ins = 0

    for sent in clean_sentences:
        sent = str(sent).strip()
        if not sent:
            continue

        # 1. Generate comma splices
        if count_splice < max_per_type:
            corrupted = make_comma_splice(sent)
            if corrupted and corrupted != sent:
                rows.append({
                    "sentence": sent,
                    "label": 0,
                    "error_type": "orig",
                    "synthetic_source": "splice"
                })
                rows.append({
                    "sentence": corrupted,
                    "label": 1,
                    "error_type": "comma_splice",
                    "synthetic_source": "splice"
                })
                count_splice += 1

        # 2. Generate comma deletions
        if count_del < max_per_type:
            corrupted = remove_random_comma(sent)
            if corrupted and corrupted != sent:
                rows.append({
                    "sentence": sent,
                    "label": 0,
                    "error_type": "orig",
                    "synthetic_source": "delete"
                })
                rows.append({
                    "sentence": corrupted,
                    "label": 1,
                    "error_type": "comma_deleted",
                    "synthetic_source": "delete"
                })
                count_del += 1

        # 3. Generate comma insertions
        if count_ins < max_per_type:
            corrupted = insert_random_comma(sent)
            if corrupted and corrupted != sent:
                rows.append({
                    "sentence": sent,
                    "label": 0,
                    "error_type": "orig",
                    "synthetic_source": "insert"
                })
                rows.append({
                    "sentence": corrupted,
                    "label": 1,
                    "error_type": "comma_inserted",
                    "synthetic_source": "insert"
                })
                count_ins += 1

        # stop early if all quotas filled
        if (
            count_splice >= max_per_type and
            count_del >= max_per_type and
            count_ins >= max_per_type
        ):
            break

    df_syn = pd.DataFrame(rows).drop_duplicates(subset=["sentence", "label"])
    df_syn = df_syn.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("Synthetic examples per type:")
    print("  comma splices:", count_splice)
    print("  comma deletions:", count_del)
    print("  comma insertions:", count_ins)
    print("Total rows in df_syn:", len(df_syn))

    return df_syn