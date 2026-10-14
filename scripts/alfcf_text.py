"""Dependency-free copies of the ALF-CF text helpers (note template, system prompt, seeds), for the vLLM env,
which has no alfworld. Identical to alfcf_common.note_text / parse_note / stable_seed and
alfcf_build_sft.SYSTEM; the pilot driver asserts SYSTEM equals the one stored in the SFT text data."""
import hashlib
import re

SYSTEM = (
    "You are a household agent in a text-based environment. Each turn, reply with one action in the form "
    "<action>COMMAND</action>. While you are searching for an object that you have not seen yet, you may first "
    "write where you expect to find it, on the line before the action, in the form <prediction>The OBJECT is most "
    "likely in a TYPE, then a TYPE, then a TYPE.</prediction>\n"
    "Commands: go to RECEPTACLE, open RECEPTACLE, close RECEPTACLE, examine RECEPTACLE, take OBJECT from "
    "RECEPTACLE, move OBJECT to RECEPTACLE, heat OBJECT with RECEPTACLE, cool OBJECT with RECEPTACLE, clean OBJECT "
    "with RECEPTACLE, use OBJECT, look, inventory.")
OBS_OPEN, OBS_CLOSE = "\n<observation>", "</observation>\n"
OBJ_RE = re.compile(r"\b(?:a|an) ([a-z]+ \d+)")


def note_text(obj, ranked_types):
    ts = list(ranked_types)[:3]
    if not ts:
        return None
    s = f"The {obj} is most likely in a {ts[0]}"
    for t in ts[1:]:
        s += f", then a {t}"
    return s + "."


def parse_note(text):
    m = re.match(r"\s*The (\S+) is most likely in a (\S+?)(?:,|\.)(.*)$", text.strip())
    if not m:
        return None
    ts = [m.group(2)] + re.findall(r"then a (\S+?)(?:,|\.|$)", m.group(3))
    return m.group(1), ts


def stable_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def receps_from_intro(intro):
    part = re.split("you see", intro, flags=re.I)[-1].split("Your task is to")[0]
    return [re.sub(r"^(and )?an? ", "", r.strip().strip(" .\n")) for r in part.split(",") if r.strip()]
