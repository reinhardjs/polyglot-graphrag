Extract a legal knowledge graph from the document below.
Entities: extract names EXACTLY as they appear — do NOT translate or paraphrase.
Relationships: use one of GOVERNS, REFERENCES, OVERRIDES, OBLIGATES, AMENDS, DEFINES.
Return ONLY valid JSON, no prose, no markdown:
{"nodes":[{"id":"ExactLegalEntityName","type":"Statute|Regulation|Clause|Party|Contract|Policy|Obligation"}],"edges":[{"source":"entity_a","target":"entity_b","type":"GOVERNS|REFERENCES|OVERRIDES|OBLIGATES|AMENDS|DEFINES"}]}
Document ({doc_id}):
{text}
