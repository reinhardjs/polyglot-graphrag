"""index_router.py — JSON object salvage utility for extraction fallbacks.

Only `_salvage_objects` is live: it recovers complete {...} JSON objects from
possibly-truncated LLM output, and is used as a last-resort parser in
hybrid_extraction.py when the model's relation output is cut off at
max_tokens. The original V3.0 Index-Routing extraction pipeline that lived
here (GLiNER → Qwen relation classification) was deprecated (20% precision)
and removed; extraction now uses hybrid_extraction.py (sliding_window /
hybrid / llm modes).
"""

import json
from typing import Dict, Any, List


def _salvage_objects(text: str) -> List[Dict[str, Any]]:
    """Extract complete {...} JSON objects from a possibly-truncated string.

    Used when the LLM output is cut off at max_tokens mid-array. Scans for
    balanced-brace objects and parses each independently, discarding any
    trailing incomplete fragment.
    """
    out = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i  # only track the outermost object's start
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    frag = text[start:i + 1]
                    try:
                        obj = json.loads(frag)
                        # Accept either a bare relation object or a wrapper
                        # {"relations":[...]} (recurse into its list).
                        if isinstance(obj, dict) and "relations" in obj:
                            for rel in obj.get("relations", []):
                                if isinstance(rel, dict) and "source" in rel \
                                        and "target" in rel:
                                    out.append(rel)
                        elif isinstance(obj, dict) and "source" in obj \
                                and "target" in obj:
                            out.append(obj)
                    except Exception:
                        pass
                    start = None
    return out
