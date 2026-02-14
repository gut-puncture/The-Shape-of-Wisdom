#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


V1_REQUIRED = [
    "model_name",
    "prompt_id",
    "example_id",
    "chosen_letter",
    "is_correct",
    "layer_candidate_probs",
    "layer_entropy",
    "layer_margin",
    "projected_hidden_128",
]

V2_REQUIRED = [
    "response_text",
    "response_token_ids",
    "response_token_pieces",
    "response_diagnostics",
]


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate inference outputs against prompt manifest")
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=False)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--expected-wrapper-count", type=int, default=0)
    ap.add_argument("--require-v2-fields", action="store_true")
    ap.add_argument(
        "--require-option-token-invariance",
        action="store_true",
        help="Require meta.candidate_scoring.kind == 'option_bucket_logsumexp' and required bucket metadata",
    )
    args = ap.parse_args()

    prompt_rows = _load_jsonl(args.prompts)
    out_rows = _load_jsonl(args.outputs)

    expected = len(prompt_rows)
    got = len(out_rows)

    required = list(V1_REQUIRED)
    if args.require_v2_fields:
        required += V2_REQUIRED

    meta_obj = None
    if args.meta and args.meta.exists():
        meta_obj = json.loads(args.meta.read_text())

    expected_proj_dim = 128
    if isinstance(meta_obj, dict):
        expected_proj_dim = int(meta_obj.get("pca_components_effective", meta_obj.get("pca_components", 128)) or 128)

    missing_key_rows = 0
    bad_prob_shape_rows = 0
    bad_logit_shape_rows = 0
    bad_projected_rows = 0
    bad_response_rows = 0
    letters_ok = 0
    letters_total = 0
    wrapper_ids = set()

    for o in out_rows:
        if any(k not in o for k in required):
            missing_key_rows += 1

        probs = o.get("layer_candidate_probs", [])
        if probs and any((not isinstance(p, list)) or len(p) != 4 for p in probs):
            bad_prob_shape_rows += 1

        logits = o.get("layer_candidate_logits", [])
        if logits and any((not isinstance(p, list)) or len(p) != 4 for p in logits):
            bad_logit_shape_rows += 1

        proj = o.get("projected_hidden", o.get("projected_hidden_128", []))
        if proj and any((not isinstance(v, list)) or len(v) != expected_proj_dim for v in proj):
            bad_projected_rows += 1

        if args.require_v2_fields:
            rtxt = o.get("response_text")
            rids = o.get("response_token_ids")
            rpieces = o.get("response_token_pieces")
            rdiag = o.get("response_diagnostics")
            if not isinstance(rtxt, str):
                bad_response_rows += 1
            if not isinstance(rids, list) or not all(isinstance(x, int) for x in rids):
                bad_response_rows += 1
            if not isinstance(rpieces, list) or not all(isinstance(x, str) for x in rpieces):
                bad_response_rows += 1
            if not isinstance(rdiag, dict):
                bad_response_rows += 1

        c = o.get("chosen_letter")
        if c is not None:
            letters_total += 1
            if c in {"A", "B", "C", "D"}:
                letters_ok += 1

        wid = o.get("wrapper_id")
        if wid is not None:
            wrapper_ids.add(wid)

    meta_ok = True
    meta_missing = []
    if args.require_option_token_invariance:
        if not isinstance(meta_obj, dict):
            meta_ok = False
            meta_missing.append("meta")
        else:
            cs = meta_obj.get("candidate_scoring")
            if not isinstance(cs, dict):
                meta_ok = False
                meta_missing.append("meta.candidate_scoring")
            else:
                if cs.get("kind") != "option_bucket_logsumexp":
                    meta_ok = False
                    meta_missing.append("meta.candidate_scoring.kind")
                for k in ["buckets", "buckets_sha256", "bucket_union_size", "normalization", "aggregation", "projection"]:
                    if k not in cs:
                        meta_ok = False
                        meta_missing.append(f"meta.candidate_scoring.{k}")

                # buckets must have A/B/C/D and non-empty.
                b = cs.get("buckets")
                if not isinstance(b, dict) or any(opt not in b for opt in ["A", "B", "C", "D"]):
                    meta_ok = False
                    meta_missing.append("meta.candidate_scoring.buckets.ABCD")
                else:
                    for opt in ["A", "B", "C", "D"]:
                        if not isinstance(b.get(opt), list) or len(b.get(opt)) == 0:
                            meta_ok = False
                            meta_missing.append(f"meta.candidate_scoring.buckets.{opt}_nonempty")

    pass_ok = (
        expected == got
        and missing_key_rows == 0
        and bad_prob_shape_rows == 0
        and bad_logit_shape_rows == 0
        and bad_projected_rows == 0
        and bad_response_rows == 0
        and meta_ok
    )

    if args.expected_wrapper_count > 0:
        pass_ok = pass_ok and (len(wrapper_ids) == args.expected_wrapper_count)

    report = {
        "expected_rows": expected,
        "output_rows": got,
        "row_count_match": expected == got,
        "missing_key_rows": missing_key_rows,
        "bad_prob_shape_rows": bad_prob_shape_rows,
        "bad_logit_shape_rows": bad_logit_shape_rows,
        "bad_projected_rows": bad_projected_rows,
        "bad_response_rows": bad_response_rows,
        "chosen_letter_valid_fraction": (letters_ok / letters_total if letters_total else 0.0),
        "wrapper_id_count": len(wrapper_ids),
        "meta_exists": bool(meta_obj is not None),
        "meta": meta_obj,
        "require_v2_fields": bool(args.require_v2_fields),
        "require_option_token_invariance": bool(args.require_option_token_invariance),
        "meta_ok": bool(meta_ok),
        "meta_missing": meta_missing,
        "pass": pass_ok,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if pass_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
