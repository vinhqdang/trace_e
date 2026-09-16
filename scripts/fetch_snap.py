"""Download SNAP graphs used in influence-maximisation papers and convert them
to the 0-indexed undirected edge-list format in data/networks/.

    python scripts/fetch_snap.py ca-GrQc ca-HepTh soc-Epinions1

Names map to https://snap.stanford.edu/data/<name>.txt.gz. Self-loops and
duplicate edges are dropped and nodes are relabelled 0..N-1 in order of first
appearance. Output files are named after the SNAP id in lower case with dashes
removed (ca-GrQc -> cagrqc, soc-Epinions1 -> socepinions1).
"""
import gzip
import io
import os
import sys
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "networks")


def convert(name: str) -> str:
    if os.path.exists(name):  # local .txt.gz already downloaded
        raw = open(name, "rb").read()
        name = os.path.basename(name).replace(".txt.gz", "")
    else:
        url = f"https://snap.stanford.edu/data/{name}.txt.gz"
        print("downloading", url)
        raw = urllib.request.urlopen(url, timeout=120).read()
    ids, edges = {}, set()
    with gzip.open(io.BytesIO(raw), "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            a, b = line.split()[:2]
            if a == b:
                continue
            u = ids.setdefault(a, len(ids))
            v = ids.setdefault(b, len(ids))
            edges.add((min(u, v), max(u, v)))
    out = os.path.join(OUT, name.lower().replace("-", "") + ".csv")
    with open(out, "w") as f:
        for u, v in sorted(edges):
            f.write(f"{u} {v}\n")
    print(f"{name}: {len(ids)} nodes, {len(edges)} edges -> {out}")
    return out


if __name__ == "__main__":
    for n in sys.argv[1:] or ["ca-GrQc", "ca-HepTh"]:
        convert(n)
