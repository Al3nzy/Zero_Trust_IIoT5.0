"""
Tamper-evident audit ledger (hash chain + Ed25519 signatures + external anchors).
This is an append-only audit log, NOT a distributed blockchain: one writer, no consensus, no smart contracts.
Trust assumptions are explicit: the writer's signing key is kept outside the log store, and anchor digests are
periodically copied to a separate witness store (WORM storage / remote syslog / trusted timestamping service).
"""
import hashlib, json, time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

GENESIS = "0" * 64


def _h(index, ts, payload, prev):
    return hashlib.sha256(f"{index}|{ts}|{payload}|{prev}".encode()).hexdigest()


class Ledger:
    def __init__(self, signer: Ed25519PrivateKey | None = None, anchor_every: int = 50, anchor_store=None):
        self.chain, self.signer, self.anchor_every = [], signer, anchor_every
        self.anchors = anchor_store if anchor_store is not None else []     # the external witness
        self.append({"event": "genesis"})

    def append(self, record: dict):
        prev = self.chain[-1]["hash"] if self.chain else GENESIS
        idx, ts = len(self.chain), round(time.time(), 6)
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
        hh = _h(idx, ts, payload, prev)
        blk = dict(index=idx, ts=ts, payload=payload, prev=prev, hash=hh)
        if self.signer is not None:
            blk["sig"] = self.signer.sign(bytes.fromhex(hh)).hex()
        self.chain.append(blk)
        if self.anchor_every and idx % self.anchor_every == 0:
            self.anchors.append((idx, hh))                                  # copy head digest to the witness
        return blk


def verify(chain, pub: Ed25519PublicKey | None = None, anchors=None, use_sig=True, use_anchor=True):
    """Return (ok, reason). Checks links/hashes; optionally signatures and anchors."""
    prev = GENESIS
    for i, b in enumerate(chain):
        if b["index"] != i or b["prev"] != prev: return False, f"link@{i}"
        if _h(b["index"], b["ts"], b["payload"], b["prev"]) != b["hash"]: return False, f"hash@{i}"
        if use_sig and pub is not None:
            try: pub.verify(bytes.fromhex(b.get("sig", "")), bytes.fromhex(b["hash"]))
            except (InvalidSignature, ValueError): return False, f"sig@{i}"
        prev = b["hash"]
    if use_anchor and anchors:
        for idx, hh in anchors:
            if idx >= len(chain) or chain[idx]["hash"] != hh: return False, f"anchor@{idx}"
    return True, "ok"
