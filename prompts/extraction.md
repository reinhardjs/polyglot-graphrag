Extract a knowledge graph from the engineering document below.
Entities: extract names EXACTLY as they appear — do NOT translate.
  (e.g. 'Basis Data' stays 'Basis Data', 'Database' stays 'Database').
Relationships: use one of ASSOCIATED_WITH, DEPENDS_ON, IMPACTS,
  AUTHORED, REFERENCES, FIXES.
Return ONLY valid JSON, no prose, no markdown:
{{"nodes":[{"id":"ExactEntityName","type":"Microservice|Database|API|Metric|Developer|Framework|Component|Bug|PR|ADR"}],"edges":[{"source":"entity_a","target":"entity_b","type":"ASSOCIATED_WITH|DEPENDS_ON|IMPACTS|AUTHORED|REFERENCES|FIXES"}]}}
Document ({doc_id}):
{text}
