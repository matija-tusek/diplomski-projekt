"""Locate LLM evidence quotes in the source text and compute character offsets.

LLMs cannot count characters, so the model only quotes and this code finds the quote.
Matching runs in two steps:

1. exact: `source.find(quote)`.
2. relaxed: compare only letters and digits, lowercased. This tolerates the small
   copying slips a 4B model makes, e.g. "German (A-2)" for "German (A2)", "machine learning"
   for "machine-learning", or changed spaces and capital letters.

On a relaxed match, the evidence text is replaced with the real slice of the source, so the
invariant `evidence.text == raw_text[char_start:char_end]` always holds. A quote that is not
found either way returns None: the model paraphrased or invented it, and the caller drops the item.
"""


def _alnum_index(text: str) -> tuple[str, list[int]]:
    """Lowercased letters/digits of `text` plus, for each one, its index in `text`."""
    chars, positions = [], []
    for i, ch in enumerate(text):
        if ch.isalnum():
            chars.append(ch.lower())
            positions.append(i)
    return "".join(chars), positions


def locate(quote: str | None, source: str, source_field: str) -> dict | None:
    """Return a contract Evidence dict for `quote`, or None if it is not in `source`."""
    if not quote or not quote.strip():
        return None
    quote = quote.strip()

    start = source.find(quote)
    if start >= 0:
        end = start + len(quote)
    else:
        needle, _ = _alnum_index(quote)
        haystack, positions = _alnum_index(source)
        hit = haystack.find(needle) if needle else -1
        if hit < 0:
            return None
        start = positions[hit]
        end = positions[hit + len(needle) - 1] + 1

    return {
        "text": source[start:end],
        "source_field": source_field,
        "char_start": start,
        "char_end": end,
    }
