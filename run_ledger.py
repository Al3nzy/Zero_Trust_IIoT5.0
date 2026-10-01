"""Stage 5: tamper-detection experiment for the audit ledger (hash chain vs +signatures vs +external anchors)."""
import sys, os, json, time, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ztids import config as C
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ztids.ledger import Ledger, verify, _h, GENESIS

rng = np.random.RandomState(0)
N, TRIALS, ANCH = (200, 20, 25) if C.QUICK else (1000, 200, 50)
key = Ed25519PrivateKey.generate(); pub = key.public_key()

def build(n=N, signer=key, anchor_every=ANCH):
    L = Ledger(signer=signer, anchor_every=anchor_every)
    for i in range(n - 1): L.append({"dev": f"d{i%10}", "T": round(float(rng.rand()), 4), "state": "Healthy"})
    return L

def rehash_from(chain, i, signer=None):
    for j in range(i, len(chain)):
        b = chain[j]; b["index"] = j; b["prev"] = chain[j - 1]["hash"] if j else GENESIS
        b["hash"] = _h(b["index"], b["ts"], b["payload"], b["prev"])
        if signer is not None: b["sig"] = signer.sign(bytes.fromhex(b["hash"])).hex()

def attack(kind, L):
    ch = copy.deepcopy(L.chain); i = int(rng.randint(1, len(ch) - 1))
    if kind == "edit_no_rehash":
        ch[i]["payload"] = ch[i]["payload"].replace("Healthy", "Quarantined")
    elif kind == "edit_rehash_no_key":
        ch[i]["payload"] = ch[i]["payload"].replace("Healthy", "Quarantined"); rehash_from(ch, i)
    elif kind == "edit_rehash_with_key":
        ch[i]["payload"] = ch[i]["payload"].replace("Healthy", "Quarantined"); rehash_from(ch, i, key)
    elif kind == "delete_relink_no_key":
        del ch[i]; rehash_from(ch, i)
    elif kind == "truncate_with_key":
        k = int(rng.randint(1, 101)); ch = ch[:-k]
    elif kind == "rewrite_own_key":
        k2 = Ed25519PrivateKey.generate(); L2 = Ledger(signer=k2, anchor_every=0)
        for b in ch[1:]: L2.append(json.loads(b["payload"]))
        ch = L2.chain
    return ch

kinds = ["edit_no_rehash", "edit_rehash_no_key", "edit_rehash_with_key", "delete_relink_no_key", "truncate_with_key", "rewrite_own_key"]
cfgs = {"hash chain only": dict(pub=None, use_sig=False, use_anchor=False),
        "+ signatures": dict(pub=pub, use_sig=True, use_anchor=False),
        "+ signatures + anchors": dict(pub=pub, use_sig=True, use_anchor=True)}
L = build()
ok0 = {c: verify(L.chain, anchors=L.anchors, **kw)[0] for c, kw in cfgs.items()}
res = {"untampered_verifies": ok0, "attacks": {}}
for k in kinds:
    det = {c: 0 for c in cfgs}
    for _ in range(TRIALS):
        ch = attack(k, L)
        for c, kw in cfgs.items():
            if not verify(ch, anchors=L.anchors, **kw)[0]: det[c] += 1
    res["attacks"][k] = {c: det[c] / TRIALS for c in cfgs}
    print(k, res["attacks"][k], flush=True)

# performance
perf = {}
for signed in (False, True):
    L2 = Ledger(signer=key if signed else None, anchor_every=ANCH); t = time.perf_counter()
    for i in range(500 if C.QUICK else 5000): L2.append({"dev": "d1", "T": 0.5, "state": "Degraded"})
    perf[f"append_us_{'signed' if signed else 'hash_only'}"] = (time.perf_counter() - t) / (500 if C.QUICK else 5000) * 1e6
for n in ((1000, 2000) if C.QUICK else (1000, 10000, 50000)):
    L3 = Ledger(signer=key, anchor_every=ANCH)
    for i in range(n - 1): L3.append({"d": i})
    t = time.perf_counter(); verify(L3.chain, pub, L3.anchors); perf[f"verify_s_{n}"] = time.perf_counter() - t
res["perf"] = perf; print(perf)
json.dump(res, open(os.path.join(C.RES, "ledger.json"), "w"), indent=1)
