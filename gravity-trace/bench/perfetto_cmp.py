"""Two Perfetto JSON (.json.gz) files of bench/perfetto_py_vs_rust.sh event by event: the multiset of each event type
(names, tracks, times, args; flow ids and the order of events left out). Prints the counts and the differences."""
import collections
import gzip
import sys

import orjson


def load(path):
    """ph -> Counter of event keys, and the X events' names separately."""
    ev = collections.defaultdict(collections.Counter)
    names = collections.Counter()
    with gzip.open(path, "rb") as f:
        for line in f:
            line = line.strip().rstrip(b",")
            if not line.startswith(b"{") or line.startswith(b'{"schemaVersion'):
                continue
            e = orjson.loads(line)
            ph = e["ph"]
            args = tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in e.get("args", {}).items()))
            if ph == "X":
                names[e["name"]] += 1
                key = (e["cat"], e["pid"], e["tid"], e["ts"], e["dur"], args)
            elif ph in ("s", "f"):
                key = (e["pid"], e["tid"], e["ts"])
            elif ph == "C":
                key = (e["name"], e["pid"], e["ts"], e["args"]["value"])
            else:
                key = (e["name"], e["pid"], e.get("tid"), args)
            ev[ph][hash(key)] += 1
    return ev, names


a, b = sys.argv[1], sys.argv[2]
(ea, na), (eb, nb) = load(a), load(b)
ok = True
for ph in sorted(set(ea) | set(eb)):
    ca, cb = ea[ph], eb[ph]
    only_a, only_b = sum((ca - cb).values()), sum((cb - ca).values())
    ok &= only_a == only_b == 0
    print(f"  ph {ph}: {sum(ca.values()):>9} vs {sum(cb.values()):>9}  differing: {only_a} / {only_b}")
dn = sum((na - nb).values())
print(f"  X names differing: {dn} events, {len(set(na) ^ set(nb))} distinct names (demangler: cpp_demangle vs c++filt)")
for n in list(set(na) - set(nb))[:2]:
    print(f"    only in {a.rsplit('/', 1)[-1]}: {n[:160]}")
for n in list(set(nb) - set(na))[:2]:
    print(f"    only in {b.rsplit('/', 1)[-1]}: {n[:160]}")
print("SAME EVENTS" if ok else "DIFFERENT")
