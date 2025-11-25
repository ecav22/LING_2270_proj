import ast
import spacy
import nltk
from nltk import CFG, ChartParser

nlp = spacy.load("en_core_web_sm")

# Helper functions for ipynb file

def parse_corrections(s):
    # Turn "['x', 'y']" into a Python list
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

def comma_change_row(sentence, corrections):
    if not isinstance(sentence, str):
        return False
    if ',' not in sentence:
        return False
    orig_commas = sentence.count(',')
    for c in corrections:
        if not isinstance(c, str):
            continue
        # 1) Fewer commas in correction
        if c.count(',') < orig_commas:
            return True
        # 2) Punctuation changed (e.g., period or semicolon)
        if ('.' in c or ';' in c) and orig_commas > 0:
            return True
    return False

#For spaCy, the POS tagger is quite complex, so we will reduce to simplied version
clause_parser = None
s_parser = None

def analyze(sentence):
    doc = nlp(sentence)
    for token in doc:
        print(token.text, token.pos_, token.tag_)


def coarse_pos(token):
    """
    Map spaCy's token.pos_ to a tiny tagset used in our CFG
    """
    if token.pos_ in ("NOUN", "PROPN"):
        return "N"
    if token.pos_ == "PRON":
        return "PRON"
    if token.pos_ in ("AUX",):
        return "AUX"
    if token.pos_ in ("VERB",):
        return "V"
    if token.pos_ == "DET":
        return "DET"
    if token.pos_ == "ADJ":
        return "ADJ"
    if token.pos_ == "ADV":
        return "ADV"
    if token.pos_ == "ADP":  # prepositions
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
    # fallback
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
    words, tags = sent_to_words_tags(sentence)

    # Must have COMMA and final PUNCT
    if "COMMA" not in tags or (len(tags) == 0 or tags[-1] != "PUNCT"):
        return False

    # If grammar already thinks full S is OK, we *don’t* call it a comma splice
    if parses_as_sentence(tags):
        return False

    # Try splitting at each COMMA in the tag sequence
    for i, tag in enumerate(tags):
        if tag != "COMMA":
            continue

        left_tags  = tags[:i]         # before comma
        right_tags = tags[i+1:-1]     # between comma and sentence-final PUNCT

        if not left_tags or not right_tags:
            continue

        if parses_as_clause(left_tags) and parses_as_clause(right_tags):
            # Clause , Clause; not licensed as S ⇒ comma splice candidate
            return True

    return False