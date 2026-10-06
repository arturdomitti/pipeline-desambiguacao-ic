import json, os

cache = ".pipeline_cache"
partial = os.path.join(cache, "step2_partial_1000_todos.jsonl")

rows = [json.loads(l) for l in open(partial, encoding="utf-8") if l.strip()]
com_cand = [i for i, r in enumerate(rows) if r["candidates"]]
print("Total:", len(rows), "| com candidatos:", len(com_cand))
print("Última posição com candidatos:", max(com_cand), "de", len(rows))