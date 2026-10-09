"""Extract only the needed members of OmniSQL data.zip via HTTP range requests.

One streamed range request per member, decompressed locally, CRC32-checked.
Writes under OUT/<original zip path>. Idempotent: skips members whose file
already exists with the right size.
"""
import concurrent.futures as cf
import json
import os
import struct
import sys
import tempfile
import zlib

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
from remote_zip import URL, open_zip  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "omnisql")


def wanted(names):
    train = pd.read_parquet(os.path.join(ROOT, "data", "skyrl_sql", "train.parquet"))
    synsql_ids = set(train["db_id"])
    keep = []
    for n in names:
        if n.endswith("/"):
            continue
        p = n.split("/")
        if n in ("data/dev_bird.json", "data/dev_spider.json", "data/spider/dev.json", "data/spider/tables.json"):
            keep.append(n)
        elif n.startswith("data/spider/database/"):
            keep.append(n)
        elif n.startswith("data/bird/dev_20240627/"):
            keep.append(n)
        elif n.startswith("data/SynSQL-2.5M/databases/") and len(p) > 3 and p[3] in synsql_ids:
            keep.append(n)
    return keep, synsql_ids


def resolve_url(session):
    return session.head(URL, allow_redirects=True, timeout=60).url


def fetch(info, session_url):
    dest = os.path.join(OUT, info.filename)
    if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
        return info.filename, "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    s = requests.Session()
    url = session_url
    for attempt in range(6):
        try:
            hdr = s.get(url, headers={"Range": f"bytes={info.header_offset}-{info.header_offset + 29}"}, timeout=120)
            if hdr.status_code in (403, 410):
                url = resolve_url(s)
                continue
            n_len, e_len = struct.unpack("<HH", hdr.content[26:30])
            start = info.header_offset + 30 + n_len + e_len
            end = start + info.compress_size - 1
            crc = 0
            dec = zlib.decompressobj(-15) if info.compress_type == 8 else None
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest))
            with os.fdopen(fd, "wb") as f:
                if info.compress_size > 0:
                    with s.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=300) as r:
                        if r.status_code in (403, 410):
                            url = resolve_url(s)
                            raise IOError("expired")
                        r.raise_for_status()
                        for chunk in r.iter_content(1 << 20):
                            data = dec.decompress(chunk) if dec else chunk
                            crc = zlib.crc32(data, crc)
                            f.write(data)
                    if dec:
                        tail = dec.flush()
                        crc = zlib.crc32(tail, crc)
                        f.write(tail)
            if (crc & 0xFFFFFFFF) != info.CRC or os.path.getsize(tmp) != info.file_size:
                os.remove(tmp)
                raise IOError(f"crc/size mismatch {info.filename}")
            os.replace(tmp, dest)
            return info.filename, "ok"
        except Exception as e:  # retry
            last = e
    return info.filename, f"FAIL {last}"


def main():
    z = open_zip()
    infos = {i.filename: i for i in z.infolist()}
    keep, synsql_ids = wanted(list(infos))
    total = sum(infos[k].file_size for k in keep)
    print(f"{len(keep)} members, {total/1e9:.2f} GB uncompressed; {len(synsql_ids)} SynSQL train db_ids", flush=True)
    url = resolve_url(requests.Session())
    results = {"ok": 0, "skip": 0, "fail": []}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(fetch, infos[k], url) for k in keep]
        for i, fu in enumerate(cf.as_completed(futs), 1):
            name, status = fu.result()
            if status == "ok":
                results["ok"] += 1
            elif status == "skip":
                results["skip"] += 1
            else:
                results["fail"].append((name, status))
            if i % 200 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} done ok={results['ok']} skip={results['skip']} fail={len(results['fail'])}", flush=True)
    found = {p.split('/')[3] for p in keep if p.startswith('data/SynSQL-2.5M/databases/')}
    missing = sorted(synsql_ids - found)
    manifest = {"members": len(keep), "bytes": total, "ok": results["ok"], "skip": results["skip"],
                "fail": results["fail"], "synsql_train_db_ids": len(synsql_ids), "synsql_missing": missing}
    with open(os.path.join(OUT, "extract_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: (v if k != "fail" else len(v)) for k, v in manifest.items() if k != "synsql_missing"}), "missing_synsql:", len(missing))


if __name__ == "__main__":
    main()
