"""Fetch publisher/DataCite BibTeX for every cited key (no hand-written entries except the SkyRL repo)."""
import json, re, subprocess, sys
ARXIV = {
 "armengol2025et": "2503.05703", "cai2026aawm": "2606.25421", "chen2022codet": "2207.10397",
 "chen2023selfdebug": "2304.05128", "chen2025spa": "2510.15047", "copet2025cwm": "2510.02387",
 "elich2023mtl": "2311.04698", "fu2024preact": "2402.11534", "gehring2024rlef": "2410.02089",
 "gong2025sqlens": "2506.04494", "gu2024webdreamer": "2411.06559", "jiang2025feedbackfriction": "2506.11930",
 "jin2025searchr1": "2503.09516", "kumar2024score": "2409.12917", "kurin2022unitary": "2201.04122",
 "lee2026solidcoder": "2604.19825", "li2023bird": "2305.03111", "li2024selectsql": "2401.02115",
 "li2025omnisql": "2503.02240", "li2026tapo": "2607.27973", "liu2026serlsql": "2608.00485",
 "lu2026paw": "2606.02388", "ni2023lever": "2302.08468", "ni2024next": "2404.14662",
 "plasticity2026": "2606.09932", "pourreza2024chase": "2410.01943", "prasad2025utgen": "2502.01619",
 "shao2024deepseekmath": "2402.03300", "shrivastava2026echo": "2605.24517", "tian2026pvsql": "2604.17653",
 "wang2018egdecoding": "1807.03100", "wang2025vagen": "2510.16907", "wang2026darkroom": "2607.21273",
 "xu2025mtirsql": "2510.25510", "yang2025qwen3": "2505.09388", "yu2018spider": "1809.08887",
 "yu2025dynamind": "2510.09577", "yu2025dynathink": "2506.00320", "yu2026rwml": "2602.05842",
 "zhang2025earlyexperience": "2510.08558", "zhang2026actobs": "2609.20715",
 "schmied2025greedy": "2504.16078", "cheng2026tooluse": "2605.14038", "orgad2024knowmore": "2410.02707",
 "zeng2025turnlevel": "2505.11821", "yue2025rlreasoning": "2504.13837"}
def curl(args):
    return subprocess.run(["curl", "-sSL", "--max-time", "40"] + args, capture_output=True, text=True).stdout
out, report = [], {}
for key, aid in ARXIV.items():
    bib = curl(["-H", "Accept: application/x-bibtex", f"https://doi.org/10.48550/arXiv.{aid}"]).strip()
    if not bib.startswith("@"):
        report[key] = "FAILED"; continue
    bib = re.sub(r"^@(\w+)\{[^,]*,", lambda m: "@" + m.group(1) + "{" + key + ",", bib, count=1)
    t = re.search(r"title\s*=\s*\{(.+?)\},?\s*\n", bib, re.S)
    report[key] = (aid, t.group(1)[:90] if t else "?")
    out.append(bib)
# Li & Roth 2002 (COLING) from the ACL Anthology
bib = curl(["https://aclanthology.org/C02-1150.bib"]).strip()
if bib.startswith("@"):
    bib = re.sub(r"^@(\w+)\{[^,]*,", lambda m: "@" + m.group(1) + "{li2002questionclassification,", bib, count=1)
    t = re.search(r"title\s*=\s*\"?\{?(.+?)[\"}],?\s*\n", bib)
    out.append(bib); report["li2002questionclassification"] = ("anthology C02-1150", t.group(1) if t else "?")
else:
    report["li2002questionclassification"] = "FAILED"
out.append("""@misc{skyrl,
  title        = {{SkyRL}: A Modular Full-stack {RL} Library for {LLMs}},
  author       = {{NovaSky-AI}},
  year         = {2025},
  howpublished = {\\url{https://github.com/NovaSky-AI/SkyRL}},
  note         = {Commit 99824f0, text-to-SQL recipe}
}""")
open("references.bib", "w").write("\n\n".join(out) + "\n")
for k, v in report.items():
    print(k, v)
