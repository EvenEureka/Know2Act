"""Zero-shot location-ranking probe (protocol fixed before computing). Offline vLLM, thinking off, temperature 0. One prompt per probe: the ALFWorld intro as the
agent sees it, then a request to rank receptacle types by how likely they contain the target.
usage: python ft4_probe.py <probes.jsonl> <model_snapshot> <tag> <out.jsonl>
"""
import json, sys, re
from vllm import LLM, SamplingParams

probes_path, snap, tag, out_path = sys.argv[1:5]
probes = [json.loads(l) for l in open(probes_path)]

TEMPLATE = ("{intro}\n\nBefore you act, predict where you will find the {obj}. Rank the types of receptacles in this "
            "room from most likely to least likely to contain a {obj}. Answer with one line: a comma-separated list of "
            "receptacle types without numbers (for example: drawer, countertop, cabinet). Include every receptacle "
            "type in the room.")

llm = LLM(model=snap, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=True, seed=0)
tok = llm.get_tokenizer()
prompts = []
for p in probes:
    msg = [{"role": "user", "content": TEMPLATE.format(intro=p["intro"].strip(), obj=p["target"])}]
    prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False))
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256, seed=0))
with open(out_path, "w") as f:
    for p, o in zip(probes, outs):
        f.write(json.dumps(dict(idx=p["idx"], model=tag, response=o.outputs[0].text)) + "\n")
print("wrote", len(outs), out_path)
