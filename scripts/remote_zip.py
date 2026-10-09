"""Read members of a remote zip over HTTP range requests (no full download).

Used to pull only the needed SQLite databases out of the 22 GB OmniSQL
data.zip on Hugging Face.
"""
import io
import sys
import zipfile

import requests

URL = "https://huggingface.co/datasets/seeklhy/OmniSQL-datasets/resolve/main/data.zip"


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, block=1 << 20):
        self.session = requests.Session()
        head = self.session.head(url, allow_redirects=True, timeout=60)
        head.raise_for_status()
        self.url = head.url  # follow CDN redirect once
        self.size = int(head.headers["Content-Length"])
        self.pos = 0
        self.block = block
        self.cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = off
        elif whence == io.SEEK_CUR:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def _get(self, start, end):
        for attempt in range(5):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code == 206:
                    return r.content
                if r.status_code in (403, 410):  # signed CDN URL expired: re-resolve
                    self.url = self.session.head(URL, allow_redirects=True, timeout=60).url
            except requests.RequestException:
                pass
        raise IOError(f"range request failed {start}-{end}")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        out = bytearray()
        start = self.pos
        while len(out) < n:
            b0 = (start // self.block) * self.block
            if b0 not in self.cache:
                if len(self.cache) > 64:
                    self.cache.clear()
                self.cache[b0] = self._get(b0, min(b0 + self.block, self.size) - 1)
            chunk = self.cache[b0][start - b0 : start - b0 + (n - len(out))]
            out += chunk
            start += len(chunk)
        self.pos = start
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_zip(url=URL):
    return zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url), buffer_size=1 << 20))


if __name__ == "__main__":
    z = open_zip()
    names = z.namelist()
    print(len(names), "members")
    tops = {}
    for info in z.infolist():
        parts = info.filename.split("/")
        key = "/".join(parts[:3])
        tops.setdefault(key, [0, 0])
        tops[key][0] += 1
        tops[key][1] += info.file_size
    for k, (c, s) in sorted(tops.items(), key=lambda kv: -kv[1][1])[: int(sys.argv[1]) if len(sys.argv) > 1 else 40]:
        print(f"{s/1e9:8.2f} GB {c:7d} files  {k}")
