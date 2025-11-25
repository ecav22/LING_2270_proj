import ast

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
