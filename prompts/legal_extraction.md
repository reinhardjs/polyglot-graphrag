Extract a legal knowledge graph from the document below.
Entities: extract names EXACTLY as they appear — do NOT translate or paraphrase.
Relationships: use one of {relation_types}.
Return ONLY valid JSON, no prose, no markdown:
{"nodes":[{"id":"ExactLegalEntityName","type":"{entity_types}"}],"edges":[{"source":"entity_a","target":"entity_b","type":"{relation_types}"}]}
Document ({doc_id}):
{text}
