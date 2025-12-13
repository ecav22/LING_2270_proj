import ast
import difflib
import random
import re

import spacy
import nltk
from nltk import CFG, ChartParser


# -------------------------
# spaCy model (loaded once)
# -------------------------
nlp = spacy.load("en_core_web_sm")


# -------------------------
# Lang-8 parsing utilities
# -------------------------
def parse_corrections(s):
    """Turn "['x', 'y']" into a Python list"""
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def comma_change_row(sentence, corrections):
    """
    Heuristic: row is comma-related if correction removes commas or
    swaps commas for terminal punctuation like '.' or ';'.
    """
    if not isinstance(sentence, str):
        return False
    if "," not in sentence:
        return False
    orig_commas = sentence.count(",")
    for c in corrections:
        if not isinstance(c, str):
            continue
        if c.count(",") < orig_commas:
            return True
        if ("." in c or ";" in c) and orig_commas > 0:
            return True
    return False


# -------------------------
# CFG hooks
# -------------------------
clause_parser = None
s_parser = None


def analyze(sentence):
    doc = nlp(sentence)
    for token in doc:
        print(token.text, token.pos_, token.tag_)


def coarse_pos(token):
    """
    Map spaCy token.pos_ to a compact tagset used in CFG experiments.
    """
    if token.pos_ in ("NOUN", "PROPN"):
        return "N"
    if token.pos_ == "PRON":
        return "PRON"
    if token.pos_ == "AUX":
        return "AUX"
    if token.pos_ == "VERB":
        return "V"
    if token.pos_ == "DET":
        return "DET"
    if token.pos_ == "ADJ":
        return "ADJ"
    if token.pos_ == "ADV":
        return "ADV"
    if token.pos_ == "ADP":
        return "P"
    if token.pos_ == "CCONJ":
        return "CONJ"
    if token.pos_ == "SCONJ":
        return "SCONJ"
    if token.pos_ == "PUNCT":
        if token.text == ",":
            return "COMMA"
        if token.text in [".", "!", "?"]:
            return "PUNCT"
    return token.pos_


def sent_to_words_tags(sentence):
    doc = nlp(sentence)
    words = [t.text for t in doc]
    tags = [coarse_pos(t) for t in doc]
    return words, tags


def parses_as_clause(tags):
    global clause_parser
    if clause_parser is None:
        raise ValueError("clause_parser has not been set. Assign it in the notebook.")
    try:
        return any(clause_parser.parse(tags))
    except ValueError:
        return False


def parses_as_sentence(tags):
    global s_parser
    if s_parser is None:
        raise ValueError("s_parser has not been set. Assign it in the notebook.")
    try:
        return any(s_parser.parse(tags))
    except ValueError:
        return False


def is_cfg_clause(tokens, tags):
    return parses_as_clause(tags)


def is_cfg_comma_splice(sentence):
    """
    POS/CFG-based comma splice detector:
    flags Clause , Clause when the whole tag sequence is NOT licensed as S.
    """
    _, tags = sent_to_words_tags(sentence)

    if "COMMA" not in tags or (len(tags) == 0 or tags[-1] != "PUNCT"):
        return False

    if parses_as_sentence(tags):
        return False

    for i, tag in enumerate(tags):
        if tag != "COMMA":
            continue

        left_tags = tags[:i]
        right_tags = tags[i + 1 : -1]

        if not left_tags or not right_tags:
            continue

        if parses_as_clause(left_tags) and parses_as_clause(right_tags):
            return True

    return False


# -------------------------
# Comma-only edit detection
# -------------------------
def tokenize(s):
    doc = nlp(s)
    return [t.text for t in doc if not t.is_space]


def get_edits(src_tokens, tgt_tokens):
    sm = difflib.SequenceMatcher(None, src_tokens, tgt_tokens)
    edits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            edits.append((tag, src_tokens[i1:i2], tgt_tokens[j1:j2]))
    return edits


def comma_only_edit(src, tgt):
    """
    True if ALL edits between src and tgt involve ONLY commas and there is at least one edit.
    """
    if src is None or tgt is None:
        return False

    src_tokens = tokenize(src)
    tgt_tokens = tokenize(tgt)
    edits = get_edits(src_tokens, tgt_tokens)

    if not edits:
        return False

    for _, old, new in edits:
        if not all(t == "," for t in old + new):
            return False

    return True


# -------------------------
# Dependency utilities
# -------------------------
def is_finite_verb(tok):
    if tok.pos_ not in ("VERB", "AUX"):
        return False

    return ("Tense" in tok.morph) or (tok.morph.get("VerbForm") == ["Fin"])


def has_subject(verb):
    return any(
        child.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass", "expl")
        for child in verb.children
    )


def extract_clause_heads(doc):
    """
    More permissive clause-head detection than "finite verb + subject child".
    We include ROOT/conj finite verbs even if subject is not a direct child.
    """
    heads = []
    for tok in doc:
        if tok.pos_ not in ("VERB", "AUX"):
            continue
        if not is_finite_verb(tok):
            continue

        if tok.dep_ in ("ROOT", "conj"):
            heads.append(tok)
            continue

        if has_subject(tok):
            heads.append(tok)

    return heads


def sentence_to_dep_symbols(text):
    """
    Convert a sentence into a compact symbol sequence:
    CLAUSE, COMMA, CONJ, PUNCT
    """
    doc = nlp(text)

    symbols = []
    clause_heads = extract_clause_heads(doc)
    clause_positions = {tok.i for tok in clause_heads}

    for tok in doc:
        if tok.i in clause_positions:
            symbols.append("CLAUSE")
        elif tok.text == ",":
            symbols.append("COMMA")
        elif tok.pos_ == "CCONJ" or tok.dep_ == "cc":
            symbols.append("CONJ")
        elif tok.text in (".", "!", "?"):
            symbols.append("PUNCT")

    return symbols


# Two parsers: one for the comma splice pattern, one for allowed patterns.
_dep_err_cfg = CFG.fromstring(
    """
ERROR -> CLAUSE COMMA CLAUSE PUNCT
"""
)
dep_err_parser = ChartParser(_dep_err_cfg)

_dep_ok_cfg = CFG.fromstring(
    """
S -> CLAUSE PUNCT
S -> CLAUSE CONJ CLAUSE PUNCT
"""
)
dep_ok_parser = ChartParser(_dep_ok_cfg)


def dep_cfg_comma_splice(sentence):
    """
    Dependency+CFG comma-splice detector:
    - symbolizes with spaCy dependencies
    - matches ERROR pattern: CLAUSE , CLAUSE .
    - excludes OK patterns: CLAUSE .  and  CLAUSE CONJ CLAUSE .
    """
    symbols = sentence_to_dep_symbols(sentence)

    if "COMMA" not in symbols or len(symbols) == 0 or symbols[-1] != "PUNCT":
        return False

    try:
        is_error = any(dep_err_parser.parse(symbols))
        if not is_error:
            return False

        is_ok = any(dep_ok_parser.parse(symbols))
        if is_ok:
            return False

        return True
    except ValueError:
        return False


# -------------------------
# ESL-aligned synthetic error generators
# -------------------------
def _safe_period(sentence: str) -> str:
    s = sentence.strip()
    if not s:
        return s
    if s[-1] not in ".!?":
        s += "."
    return s


def make_comma_splice(sentence: str):
    """
    Create a true comma splice by transforming:
      'Clause and Clause.' -> 'Clause, Clause.'
    Requires at least 2 clause heads and a coordinating conjunction.
    """
    s = _safe_period(sentence)
    doc = nlp(s)

    # Require multiple clauses
    if len(extract_clause_heads(doc)) < 2:
        return None

    # Find coordinating conjunction token
    for tok in doc:
        if tok.pos_ == "CCONJ" and tok.text.lower() in ("and", "but", "or", "so"):
            # Replace first occurrence of " <conj> " with ", "
            pattern = r"\s+" + re.escape(tok.text) + r"\s+"
            out = re.sub(pattern, ", ", s, count=1)
            if out != s:
                return out
            return None

    return None


def delete_intro_comma(sentence: str):
    """
    Remove a comma after an introductory phrase/clause if it exists.
    E.g., 'After dinner, I went home.' -> 'After dinner I went home.'
    """
    s = _safe_period(sentence)
    idx = s.find(",")
    if idx == -1 or idx > 50:
        return None

    left = s[: idx + 1]
    doc = nlp(left)
    if len(doc) < 2:
        return None

    first = doc[0].text.lower()
    if doc[0].pos_ in ("ADP", "SCONJ", "ADV") or first in (
        "after",
        "before",
        "when",
        "while",
        "although",
        "because",
        "if",
        "since",
        "as",
        "once",
    ):
        return s[:idx] + s[idx + 1 :]

    return None


def delete_comma_before_conj(sentence: str):
    """
    Remove comma before a coordinating conjunction joining independent clauses.
    E.g., 'I went home, and I slept.' -> 'I went home and I slept.'
    """
    s = _safe_period(sentence)
    m = re.search(r",\s+(and|but|or|so)\b", s, flags=re.I)
    if not m:
        return None

    doc = nlp(s)
    if len(extract_clause_heads(doc)) < 2:
        return None

    # remove the comma at m.start()
    start = m.start()
    out = s[:start] + s[start + 1 :]
    return out if out != s else None


def insert_comma_subj_verb(sentence: str):
    """
    Insert an unnecessary comma between subject and main verb:
      'The reason is clear.' -> 'The reason, is clear.'
    """
    s = _safe_period(sentence)
    doc = nlp(s)

    root = next((t for t in doc if t.dep_ == "ROOT" and t.pos_ in ("VERB", "AUX")), None)
    if root is None:
        return None

    subj = next(
        (c for c in root.children if c.dep_ in ("nsubj", "nsubjpass", "expl")),
        None,
    )
    if subj is None:
        return None

    subj_end = subj.right_edge.i
    if subj_end + 1 < len(doc) and doc[subj_end + 1].text in (",", ";", ":"):
        return None

    toks = [t.text for t in doc]
    toks.insert(subj_end + 1, ",")
    out = " ".join(toks).replace(" ,", ",")
    return out if out != s else None


def build_synthetic_esl_like(sentences, n_per_type=3000, seed=42):
    """
    Build paired (clean, corrupted) examples using ESL-aligned comma mistakes.

    Returns: list of dicts (or you can wrap with pd.DataFrame in the notebook)
      - sentence: text
      - label: 0/1
      - error_type: orig or specific error family
      - synthetic_source: "esl_like"
    """
    random.seed(seed)

    generators = [
        ("comma_splice", make_comma_splice),
        ("missing_intro_comma", delete_intro_comma),
        ("missing_comma_before_conj", delete_comma_before_conj),
        ("unnecessary_subj_verb_comma", insert_comma_subj_verb),
    ]

    rows = []
    for err_type, fn in generators:
        made = 0
        tries = 0
        # generous try budget because not every sentence supports every transformation
        while made < n_per_type and tries < n_per_type * 80:
            tries += 1
            s = random.choice(sentences)
            s_bad = fn(s)
            if s_bad and s_bad != s:
                rows.append(
                    {
                        "sentence": _safe_period(s),
                        "label": 0,
                        "error_type": "orig",
                        "synthetic_source": "esl_like",
                    }
                )
                rows.append(
                    {
                        "sentence": s_bad,
                        "label": 1,
                        "error_type": err_type,
                        "synthetic_source": "esl_like",
                    }
                )
                made += 1

    return rows