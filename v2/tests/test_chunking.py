"""Unit tests for pluggable chunking strategies (v2.6.0 REQ-3).

Pure, no GPU / daemon needed. Run:
    cd /mnt/data-970-plus/rag-system/v2
    /mnt/data-970-plus/rag-env/bin/python -m pytest tests/test_chunking.py -q
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking import (chunk_sentence, chunk_paragraph, chunk_section,
                      chunk_fixed, chunk_text, CHUNK_STRATEGIES)


def test_sentence_two():
    # Acceptance: "The cat sat. The dog ran." → 2 chunks (1 sentence / chunk)
    out = chunk_sentence("The cat sat. The dog ran.")
    assert len(out) == 2, out
    assert out[0].startswith("The cat")
    assert out[1].startswith("The dog")


def test_paragraph_three():
    text = "Para one line.\n\nPara two line.\n\nPara three line."
    out = chunk_paragraph(text)
    # 3 paragraphs → 3 chunks (1 paragraph / chunk)
    assert len(out) == 3, out


def test_section_four():
    text = ("## H1\nbody one\n\n## H2\nbody two\n\n### H3\nbody three\n\n## H4\nbody four")
    out = chunk_section(text)
    # 4 header sections → 4 chunks, each preserves its header
    assert len(out) == 4, out
    assert all(o.split("\n")[0].startswith("#") for o in out)
    assert out[0].startswith("## H1")
    assert out[2].startswith("### H3")


def test_fixed_window():
    # 1000-char text, size=500 tok (=2000 chars) → fits in ONE window
    text = "x" * 1000
    out = chunk_fixed(text, chunk_size=500, overlap=100)
    assert len(out) == 1, len(out)


def test_fixed_multichunk():
    # 6000 chars, size=500 tok (=2000 chars), step=(500-100)*4=1600
    text = "a" * 6000
    out = chunk_fixed(text, chunk_size=500, overlap=100)
    # windows at 0,1600,3200,4800 → 4 windows (last covers remainder)
    assert len(out) == 4, len(out)
    # overlap: window[1] starts 1600 in, so it shares tail of window[0]
    assert text[1600:1620] in out[1]


def test_section_preserves_header_on_subchunks():
    # A very large section should be sub-chunked but keep its header.
    big = "## Big\n" + "Sentence fragment number. " * 400
    out = chunk_section(big, chunk_size=100, overlap=20)
    assert len(out) >= 2
    assert all(o.startswith("## Big") for o in out)


def test_dispatch_unknown_falls_back_to_sentence():
    out = chunk_text("Alpha. Beta.", strategy="nonsense")
    assert len(out) == 2


def test_empty_text():
    assert chunk_sentence("") == []
    assert chunk_paragraph("   \n  ") == []
    assert chunk_section("") == []
    assert chunk_fixed("") == []


def test_all_four_registered():
    assert set(CHUNK_STRATEGIES) == {"sentence", "paragraph", "section", "fixed"}


def test_sentence_oversized_subsplit():
    # A single 2000-char sentence exceeds default 512-tok budget → sub-split.
    long_sent = "Word " * 800  # ~4000 chars → > 512 tokens
    out = chunk_sentence(long_sent, chunk_size=512, overlap=64)
    assert len(out) >= 2
    assert all(_est_ok(c, 512) for c in out)


def _est_ok(text, chunk_size):
    return round(len(text) / 4) <= chunk_size * 1.1 + 8
