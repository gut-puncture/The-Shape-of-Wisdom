#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import yaml

from _common import base_parser, run_v2_root_for, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER_PATH = REPO_ROOT / "docs" / "audit" / "V3_REQUIREMENT_LEDGER.yaml"
ALLOWED_CONDITION_TYPES = {"json_equals", "json_gte", "json_len_eq", "json_len_gte", "self_contract"}


def _load_json_obj(path: Path) -> Tuple[Any, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, str(exc)


def _json_path_get(obj: Any, path: str) -> Tuple[bool, Any]:
    p = str(path or "").strip()
    if p in {"", "$"}:
        return True, obj
    cur = obj
    for part in p.split("."):
        if isinstance(cur, Mapping):
            if part not in cur:
                return False, None
            cur = cur[part]
            continue
        if isinstance(cur, list):
            if (not str(part).isdigit()) or int(part) < 0 or int(part) >= len(cur):
                return False, None
            cur = cur[int(part)]
            continue
        return False, None
    return True, cur


def _control_passes(path: Path) -> Tuple[bool, Dict[str, Any] | None]:
    if (not path.exists()) or path.suffix.lower() != ".json":
        return False, None
    obj, err = _load_json_obj(path)
    if err is not None or (not isinstance(obj, dict)):
        return False, None
    if not bool(obj.get("pass")):
        return False, obj
    return True, obj


def _as_float(x: Any) -> float:
    if isinstance(x, bool):
        return float(int(x))
    return float(x)


def _evaluate_condition(
    *,
    cond: Mapping[str, Any],
    req_id: str,
    out_root: Path,
    json_cache: Dict[str, Any],
    virtual_json: Mapping[str, Any],
    pre_self_failing_ids: List[str],
) -> Dict[str, Any]:
    ctype = str(cond.get("type") or "")
    if ctype not in ALLOWED_CONDITION_TYPES:
        return {
            "pass": False,
            "schema_error": f"unknown_condition_type:{ctype}",
            "condition": dict(cond),
        }

    if ctype == "self_contract":
        return {
            "pass": len(pre_self_failing_ids) == 0,
            "schema_error": None,
            "condition": dict(cond),
        }

    source_rel = str(cond.get("source") or "").strip()
    if not source_rel:
        return {
            "pass": False,
            "schema_error": "missing_source",
            "condition": dict(cond),
        }

    if source_rel in virtual_json:
        src_obj = virtual_json[source_rel]
    else:
        source_path = out_root / source_rel
        if source_path.suffix.lower() != ".json":
            return {
                "pass": False,
                "schema_error": f"source_not_json:{source_rel}",
                "condition": dict(cond),
            }
        if source_rel in json_cache:
            src_obj = json_cache[source_rel]
        else:
            if not source_path.exists():
                return {
                    "pass": False,
                    "schema_error": f"source_missing:{source_rel}",
                    "condition": dict(cond),
                }
            src_obj, err = _load_json_obj(source_path)
            if err is not None:
                return {
                    "pass": False,
                    "schema_error": f"source_invalid_json:{source_rel}",
                    "condition": dict(cond),
                }
            json_cache[source_rel] = src_obj

    ok, got = _json_path_get(src_obj, str(cond.get("path") or ""))
    if not ok:
        return {
            "pass": False,
            "schema_error": None,
            "condition": dict(cond),
        }

    if ctype == "json_equals":
        return {
            "pass": got == cond.get("value"),
            "schema_error": None,
            "condition": dict(cond),
        }

    if ctype == "json_gte":
        try:
            pass_flag = _as_float(got) >= _as_float(cond.get("value"))
        except Exception:
            return {
                "pass": False,
                "schema_error": "json_gte_non_numeric",
                "condition": dict(cond),
            }
        return {
            "pass": bool(pass_flag),
            "schema_error": None,
            "condition": dict(cond),
        }

    if ctype == "json_len_eq":
        try:
            pass_flag = len(got) == int(cond.get("value"))
        except Exception:
            return {
                "pass": False,
                "schema_error": "json_len_eq_invalid",
                "condition": dict(cond),
            }
        return {
            "pass": bool(pass_flag),
            "schema_error": None,
            "condition": dict(cond),
        }

    if ctype == "json_len_gte":
        try:
            pass_flag = len(got) >= int(cond.get("value"))
        except Exception:
            return {
                "pass": False,
                "schema_error": "json_len_gte_invalid",
                "condition": dict(cond),
            }
        return {
            "pass": bool(pass_flag),
            "schema_error": None,
            "condition": dict(cond),
        }

    return {
        "pass": False,
        "schema_error": f"unhandled_condition_type:{ctype}",
        "condition": dict(cond),
    }


def _evaluate_requirement(
    *,
    req: Mapping[str, Any],
    out_root: Path,
    json_cache: Dict[str, Any],
    virtual_artifacts: set[str] | None = None,
    virtual_controls: Mapping[str, Any] | None = None,
    pre_self_failing_ids: List[str] | None = None,
) -> Dict[str, Any]:
    virtual_artifacts = set(virtual_artifacts or set())
    virtual_controls = dict(virtual_controls or {})
    pre_self_failing_ids = list(pre_self_failing_ids or [])

    req_id = str((req or {}).get("requirement_id") or "")
    claim_type = str((req or {}).get("claim_type") or "")
    artifacts = [str(x) for x in ((req or {}).get("required_artifacts") or [])]
    controls = [str(x) for x in ((req or {}).get("required_controls") or [])]

    evidence_checked: List[str] = []

    missing_artifacts: List[str] = []
    for rel in artifacts:
        evidence_checked.append(rel)
        if rel in virtual_artifacts:
            continue
        if not (out_root / rel).exists():
            missing_artifacts.append(rel)

    failing_controls: List[str] = []
    for rel in controls:
        evidence_checked.append(rel)
        if rel in virtual_controls:
            json_cache[rel] = virtual_controls[rel]
            continue
        ok, obj = _control_passes(out_root / rel)
        if not ok:
            failing_controls.append(rel)
        if obj is not None:
            json_cache[rel] = obj

    schema_errors: List[str] = []
    failing_conditions: List[Dict[str, Any]] = []

    pass_conditions = (req or {}).get("pass_conditions")
    all_of = None
    if not isinstance(pass_conditions, Mapping):
        schema_errors.append("pass_conditions_missing_or_invalid")
    else:
        all_of = pass_conditions.get("all_of")
        if not isinstance(all_of, list):
            schema_errors.append("pass_conditions_all_of_missing_or_invalid")

    condition_results: List[Dict[str, Any]] = []
    if isinstance(all_of, list):
        for cond in all_of:
            if not isinstance(cond, Mapping):
                schema_errors.append("condition_not_object")
                continue
            cond_obj = dict(cond)
            if not str(cond_obj.get("source") or "").strip() and len(controls) == 1:
                cond_obj["source"] = str(controls[0])
            if "source" in cond_obj:
                evidence_checked.append(str(cond_obj.get("source") or ""))
            cres = _evaluate_condition(
                cond=cond_obj,
                req_id=req_id,
                out_root=out_root,
                json_cache=json_cache,
                virtual_json=virtual_controls,
                pre_self_failing_ids=pre_self_failing_ids,
            )
            condition_results.append(cres)
            if cres.get("schema_error"):
                schema_errors.append(str(cres.get("schema_error")))
            if not bool(cres.get("pass")):
                failing_conditions.append(dict(cres))

    req_pass = (
        bool(req_id)
        and len(missing_artifacts) == 0
        and len(failing_controls) == 0
        and len(schema_errors) == 0
        and len(failing_conditions) == 0
    )

    return {
        "requirement_id": req_id,
        "claim_type": claim_type,
        "pass": bool(req_pass),
        "missing_artifacts": sorted(set(missing_artifacts)),
        "failing_controls": sorted(set(failing_controls)),
        "failing_conditions": failing_conditions,
        "schema_errors": sorted(set(schema_errors)),
        "evidence_checked": sorted(set(x for x in evidence_checked if x)),
    }


def main() -> int:
    ap = base_parser("V2: readiness audit against requirement ledger")
    ap.add_argument("--ledger-path", default=str(DEFAULT_LEDGER_PATH))
    args = ap.parse_args()

    out_root = run_v2_root_for(args.run_id)
    audit_path = out_root / "meta" / "readiness_audit.json"
    ledger_path = Path(args.ledger_path).expanduser().resolve()

    if not ledger_path.exists():
        payload = {
            "run_id": str(args.run_id),
            "ledger_path": str(ledger_path),
            "pass": False,
            "verdict": "NO-GO",
            "failing_requirements": ["LEDGER_MISSING"],
            "requirements": [],
            "error": f"missing ledger: {ledger_path}",
        }
        write_json(audit_path, payload)
        print(str(audit_path))
        return 2

    try:
        ledger = yaml.safe_load(ledger_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        payload = {
            "run_id": str(args.run_id),
            "ledger_path": str(ledger_path),
            "pass": False,
            "verdict": "NO-GO",
            "failing_requirements": ["LEDGER_INVALID"],
            "requirements": [],
            "error": f"invalid ledger yaml: {exc}",
        }
        write_json(audit_path, payload)
        print(str(audit_path))
        return 2

    requirements = [r for r in (ledger.get("requirements") or []) if isinstance(r, Mapping)]
    json_cache: Dict[str, Any] = {}

    non_self_results: List[Dict[str, Any]] = []
    self_reqs: List[Mapping[str, Any]] = []
    for req in requirements:
        if str(req.get("requirement_id") or "") == "RQ-014":
            self_reqs.append(req)
            continue
        non_self_results.append(
            _evaluate_requirement(
                req=req,
                out_root=out_root,
                json_cache=json_cache,
            )
        )

    pre_self_failing = sorted(
        [str(r.get("requirement_id") or "") for r in non_self_results if not bool(r.get("pass")) and str(r.get("requirement_id") or "")]
    )

    provisional = {
        "run_id": str(args.run_id),
        "ledger_path": str(ledger_path),
        "pass": len(pre_self_failing) == 0 and len(non_self_results) > 0,
        "verdict": "GO" if (len(pre_self_failing) == 0 and len(non_self_results) > 0) else "NO-GO",
        "failing_requirements": pre_self_failing,
        "requirements": non_self_results,
    }

    self_results: List[Dict[str, Any]] = []
    for req in self_reqs:
        self_results.append(
            _evaluate_requirement(
                req=req,
                out_root=out_root,
                json_cache=json_cache,
                virtual_artifacts={"meta/readiness_audit.json"},
                virtual_controls={"meta/readiness_audit.json": provisional},
                pre_self_failing_ids=pre_self_failing,
            )
        )

    result_by_id: Dict[str, Dict[str, Any]] = {
        str(r.get("requirement_id") or ""): r for r in [*non_self_results, *self_results]
    }
    req_results: List[Dict[str, Any]] = []
    for req in requirements:
        rid = str(req.get("requirement_id") or "")
        if rid in result_by_id:
            req_results.append(result_by_id[rid])

    failing = sorted(
        [str(r.get("requirement_id") or "") for r in req_results if not bool(r.get("pass")) and str(r.get("requirement_id") or "")]
    )
    verdict = "GO" if (len(req_results) > 0 and len(failing) == 0) else "NO-GO"
    payload = {
        "run_id": str(args.run_id),
        "ledger_path": str(ledger_path),
        "pass": bool(verdict == "GO"),
        "verdict": verdict,
        "failing_requirements": failing,
        "requirements": req_results,
    }
    write_json(audit_path, payload)
    print(str(audit_path))
    return 0 if verdict == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
