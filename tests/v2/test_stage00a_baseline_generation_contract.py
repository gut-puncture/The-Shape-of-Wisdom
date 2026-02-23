import importlib.util
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_00A = REPO_ROOT / "scripts" / "v2" / "00a_generate_baseline_outputs.py"
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.io_jsonl import iter_jsonl  # noqa: E402
from sow.v2.baseline_inference import resume_key_for, validate_baseline_rows  # noqa: E402


def _load_stage00a_module():
    if str(SCRIPT_00A.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_00A.parent))
    spec = importlib.util.spec_from_file_location("sow_v2_stage00a", SCRIPT_00A)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module: {SCRIPT_00A}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStage00aBaselineGenerationContract(unittest.TestCase):
    def test_validate_baseline_rows_accepts_complete_schema(self) -> None:
        row = {
            "run_id": "r",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": "rev",
            "prompt_uid": "u0",
            "example_id": "e0",
            "wrapper_id": "plain_exam",
            "coarse_domain": "biology",
            "resume_key": resume_key_for(model_id="Qwen/Qwen2.5-7B-Instruct", prompt_uid="u0"),
            "generated_text": "A",
            "first_generated_token_text": "A",
            "parsed_choice": "A",
            "parser_status": "resolved",
            "parser_signals": {"decision": "resolved_letter_first_token"},
            "is_correct": True,
            "layerwise": [
                {
                    "layer_index": 0,
                    "candidate_logits": {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
                    "candidate_probs": {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1},
                    "candidate_entropy": 0.8,
                    "top_candidate": "A",
                    "top2_margin_prob": 0.6,
                    "projected_hidden_128": [0.0] * 128,
                }
            ],
        }
        report = validate_baseline_rows(
            rows=[row],
            expected_model_id="Qwen/Qwen2.5-7B-Instruct",
            expected_model_revision="rev",
        )
        self.assertTrue(bool(report.get("pass")), msg=report)

    def test_validate_baseline_rows_fails_on_duplicate_resume_key(self) -> None:
        rk = resume_key_for(model_id="Qwen/Qwen2.5-7B-Instruct", prompt_uid="u0")
        rows = []
        for uid in ["u0", "u1"]:
            rows.append(
                {
                    "run_id": "r",
                    "model_id": "Qwen/Qwen2.5-7B-Instruct",
                    "model_revision": "rev",
                    "prompt_uid": uid,
                    "example_id": f"e_{uid}",
                    "wrapper_id": "plain_exam",
                    "coarse_domain": "biology",
                    "resume_key": rk,
                    "generated_text": "A",
                    "first_generated_token_text": "A",
                    "parsed_choice": "A",
                    "parser_status": "resolved",
                    "parser_signals": {"decision": "resolved_letter_first_token"},
                    "is_correct": True,
                    "layerwise": [
                        {
                            "layer_index": 0,
                            "candidate_logits": {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
                            "candidate_probs": {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1},
                            "candidate_entropy": 0.8,
                            "top_candidate": "A",
                            "top2_margin_prob": 0.6,
                            "projected_hidden_128": [0.0] * 128,
                        }
                    ],
                }
            )
        report = validate_baseline_rows(
            rows=rows,
            expected_model_id="Qwen/Qwen2.5-7B-Instruct",
            expected_model_revision="rev",
        )
        self.assertFalse(bool(report.get("pass")), msg=report)
        self.assertIn("duplicate_resume_key", set(report.get("errors") or []))

    def test_validate_baseline_rows_fails_on_invalid_resume_key(self) -> None:
        row = {
            "run_id": "r",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": "rev",
            "prompt_uid": "u0",
            "example_id": "e0",
            "wrapper_id": "plain_exam",
            "coarse_domain": "biology",
            "resume_key": "invalid_resume_key",
            "generated_text": "A",
            "first_generated_token_text": "A",
            "parsed_choice": "A",
            "parser_status": "resolved",
            "parser_signals": {"decision": "resolved_letter_first_token"},
            "is_correct": True,
            "layerwise": [
                {
                    "layer_index": 0,
                    "candidate_logits": {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
                    "candidate_probs": {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1},
                    "candidate_entropy": 0.8,
                    "top_candidate": "A",
                    "top2_margin_prob": 0.6,
                    "projected_hidden_128": [0.0] * 128,
                }
            ],
        }
        report = validate_baseline_rows(
            rows=[row],
            expected_model_id="Qwen/Qwen2.5-7B-Instruct",
            expected_model_revision="rev",
        )
        self.assertFalse(bool(report.get("pass")), msg=report)
        self.assertIn("invalid_resume_key", set(report.get("errors") or []))

    def test_validate_baseline_rows_fails_on_missing_resume_key(self) -> None:
        row = {
            "run_id": "r",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": "rev",
            "prompt_uid": "u0",
            "example_id": "e0",
            "wrapper_id": "plain_exam",
            "coarse_domain": "biology",
            "resume_key": "",
            "generated_text": "A",
            "first_generated_token_text": "A",
            "parsed_choice": "A",
            "parser_status": "resolved",
            "parser_signals": {"decision": "resolved_letter_first_token"},
            "is_correct": True,
            "layerwise": [
                {
                    "layer_index": 0,
                    "candidate_logits": {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
                    "candidate_probs": {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1},
                    "candidate_entropy": 0.8,
                    "top_candidate": "A",
                    "top2_margin_prob": 0.6,
                    "projected_hidden_128": [0.0] * 128,
                }
            ],
        }
        report = validate_baseline_rows(
            rows=[row],
            expected_model_id="Qwen/Qwen2.5-7B-Instruct",
            expected_model_revision="rev",
        )
        self.assertFalse(bool(report.get("pass")), msg=report)
        self.assertIn("missing_resume_key", set(report.get("errors") or []))

    def test_stage00a_writes_report_contract_with_mocked_runner(self) -> None:
        mod = _load_stage00a_module()
        run_id = "test_v2_stage00a_report_contract"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)
        try:
            cfg_path = run_root / "cfg.yaml"
            run_root.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "data_scope:",
                        "  baseline_manifest_source: /Users/shaileshrana/shape-of-wisdom/prompt_packs/ccc_baseline_v1_3000.jsonl",
                        "  baseline_manifest_sha256: bfe2557316cb1e0eae6a684eb8de84885f74e446ac72f1847a7da80baf2de56c",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            def _fake_run_baseline_for_model(*, run_id, model, manifest_rows, out_path, **kwargs):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "model_id": str(model["model_id"]),
                            "model_revision": str(model["revision"]),
                            "prompt_uid": "u0",
                            "example_id": "e0",
                            "wrapper_id": "plain_exam",
                            "coarse_domain": "biology",
                            "resume_key": resume_key_for(model_id=str(model["model_id"]), prompt_uid="u0"),
                            "generated_text": "A",
                            "first_generated_token_text": "A",
                            "parsed_choice": "A",
                            "parser_status": "resolved",
                            "parser_signals": {"decision": "resolved_letter_first_token"},
                            "is_correct": True,
                            "layerwise": [
                                {
                                    "layer_index": 0,
                                    "candidate_logits": {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
                                    "candidate_probs": {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1},
                                    "candidate_entropy": 0.8,
                                    "top_candidate": "A",
                                    "top2_margin_prob": 0.6,
                                    "projected_hidden_128": [0.0] * 128,
                                }
                            ],
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return {
                    "pass": True,
                    "stopped_early": False,
                    "rows_written": 1,
                    "output_path": str(out_path),
                    "rows_per_second": 5.0,
                    "batch_sizes_used": [4],
                }

            with patch.object(mod, "run_baseline_for_model", side_effect=_fake_run_baseline_for_model), patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT_00A),
                    "--run-id",
                    run_id,
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                    "--max-prompts",
                    "1",
                ],
            ):
                rc = mod.main()

            self.assertEqual(rc, 0)
            report_path = run_root / "v2" / "00a_generate_baseline_outputs.report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(bool(report.get("pass")), msg=report)
            self.assertIn("stats_per_model", report)
            self.assertIn("gates", report)
            self.assertIn("manifest_nonempty", set((report.get("gates") or {}).keys()))
            self.assertIn("rows_complete_per_model", set((report.get("gates") or {}).keys()))
            self.assertIn("layer_index_contract_transformer_only", set((report.get("gates") or {}).keys()))
            self.assertIn("rows_expected_full", set((report.get("manifest") or {}).keys()))
            model_stats = (report.get("stats_per_model") or {}).get("Qwen/Qwen2.5-7B-Instruct") or {}
            self.assertIn("resume_rows_total", model_stats)
            self.assertIn("resume_rows_skipped", model_stats)
            self.assertIn("checkpoint_flush_count", model_stats)
            self.assertIsNotNone(report.get("done_sentinel"))
            self.assertEqual(report.get("failing_gates"), [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage00a_fails_closed_on_existing_manifest_hash_mismatch(self) -> None:
        mod = _load_stage00a_module()
        run_id = "test_v2_stage00a_manifest_hash_mismatch"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)
        try:
            cfg_path = run_root / "cfg.yaml"
            (run_root / "manifests").mkdir(parents=True, exist_ok=True)
            (run_root / "manifests" / "ccc_baseline.jsonl").write_text(
                '{"prompt_uid":"bad","correct_key":"A","prompt_text":"x"}\n',
                encoding="utf-8",
            )
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "data_scope:",
                        "  baseline_manifest_source: /Users/shaileshrana/shape-of-wisdom/prompt_packs/ccc_baseline_v1_3000.jsonl",
                        "  baseline_manifest_sha256: bfe2557316cb1e0eae6a684eb8de84885f74e446ac72f1847a7da80baf2de56c",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(mod, "run_baseline_for_model") as mocked_runner, patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT_00A),
                    "--run-id",
                    run_id,
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                ],
            ):
                rc = mod.main()

            self.assertEqual(rc, 2)
            mocked_runner.assert_not_called()
            report_path = run_root / "v2" / "00a_generate_baseline_outputs.report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertIn("manifest_hash_contract", set(report.get("failing_gates") or []))
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage00a_fails_closed_when_manifest_source_is_missing(self) -> None:
        mod = _load_stage00a_module()
        run_id = "test_v2_stage00a_missing_manifest_source"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)
        try:
            cfg_path = run_root / "cfg.yaml"
            run_root.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "data_scope:",
                        "  baseline_manifest_source: /nonexistent/path/ccc_baseline_v1_3000.jsonl",
                        "  baseline_manifest_sha256: bfe2557316cb1e0eae6a684eb8de84885f74e446ac72f1847a7da80baf2de56c",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(mod, "run_baseline_for_model") as mocked_runner, patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT_00A),
                    "--run-id",
                    run_id,
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                ],
            ):
                rc = mod.main()

            self.assertEqual(rc, 2)
            mocked_runner.assert_not_called()
            report = json.loads((run_root / "v2" / "00a_generate_baseline_outputs.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertIn("manifest_hash_contract", set(report.get("failing_gates") or []))
            manifest = report.get("manifest") or {}
            self.assertIn("missing", str(manifest.get("hash_error") or "").lower())
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage00a_fails_closed_when_manifest_source_is_directory(self) -> None:
        mod = _load_stage00a_module()
        run_id = "test_v2_stage00a_manifest_source_directory"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)
        try:
            cfg_path = run_root / "cfg.yaml"
            manifest_dir = run_root / "fake_manifest_dir"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "data_scope:",
                        f"  baseline_manifest_source: {manifest_dir}",
                        "  baseline_manifest_sha256: bfe2557316cb1e0eae6a684eb8de84885f74e446ac72f1847a7da80baf2de56c",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(mod, "run_baseline_for_model") as mocked_runner, patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT_00A),
                    "--run-id",
                    run_id,
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                ],
            ):
                rc = mod.main()

            self.assertEqual(rc, 2)
            mocked_runner.assert_not_called()
            report = json.loads((run_root / "v2" / "00a_generate_baseline_outputs.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            manifest = report.get("manifest") or {}
            self.assertIn("file", str(manifest.get("hash_error") or "").lower())
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage00a_fails_closed_when_manifest_source_hash_mismatches(self) -> None:
        mod = _load_stage00a_module()
        run_id = "test_v2_stage00a_manifest_source_hash_mismatch"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)
        try:
            cfg_path = run_root / "cfg.yaml"
            source_manifest = run_root / "source_manifest.jsonl"
            run_root.mkdir(parents=True, exist_ok=True)
            source_manifest.write_text(
                json.dumps({"prompt_uid": "u0", "prompt_text": "Q", "correct_key": "A"}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "data_scope:",
                        f"  baseline_manifest_source: {source_manifest}",
                        "  baseline_manifest_sha256: deadbeef",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(mod, "run_baseline_for_model") as mocked_runner, patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT_00A),
                    "--run-id",
                    run_id,
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                ],
            ):
                rc = mod.main()

            self.assertEqual(rc, 2)
            mocked_runner.assert_not_called()
            report = json.loads((run_root / "v2" / "00a_generate_baseline_outputs.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            manifest = report.get("manifest") or {}
            self.assertIn("mismatch", str(manifest.get("hash_error") or "").lower())
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage00a_non_resume_clears_existing_output_before_run(self) -> None:
        mod = _load_stage00a_module()
        run_id = "test_v2_stage00a_non_resume_clears_output"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)
        try:
            cfg_path = run_root / "cfg.yaml"
            out_path = run_root / "outputs" / "Qwen__Qwen2.5-7B-Instruct" / "baseline_outputs.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            stale_row = {
                "run_id": run_id,
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "model_revision": "r0",
                "prompt_uid": "stale_u0",
                "example_id": "stale_e0",
                "wrapper_id": "plain_exam",
                "coarse_domain": "biology",
                "resume_key": resume_key_for(model_id="Qwen/Qwen2.5-7B-Instruct", prompt_uid="stale_u0"),
                "generated_text": "A",
                "first_generated_token_text": "A",
                "parsed_choice": "A",
                "parser_status": "resolved",
                "parser_signals": {"decision": "resolved_letter_first_token"},
                "is_correct": True,
                "layerwise": [
                    {
                        "layer_index": 0,
                        "candidate_logits": {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
                        "candidate_probs": {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1},
                        "candidate_entropy": 0.8,
                        "top_candidate": "A",
                        "top2_margin_prob": 0.6,
                        "projected_hidden_128": [0.0] * 128,
                    }
                ],
            }
            out_path.write_text(json.dumps(stale_row, sort_keys=True) + "\n", encoding="utf-8")

            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "data_scope:",
                        "  baseline_manifest_source: /Users/shaileshrana/shape-of-wisdom/prompt_packs/ccc_baseline_v1_3000.jsonl",
                        "  baseline_manifest_sha256: bfe2557316cb1e0eae6a684eb8de84885f74e446ac72f1847a7da80baf2de56c",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            def _fake_run_baseline_for_model(*, run_id, model, out_path, **kwargs):
                new_row = {
                    "run_id": run_id,
                    "model_id": str(model["model_id"]),
                    "model_revision": str(model["revision"]),
                    "prompt_uid": "fresh_u1",
                    "example_id": "fresh_e1",
                    "wrapper_id": "plain_exam",
                    "coarse_domain": "biology",
                    "resume_key": resume_key_for(model_id=str(model["model_id"]), prompt_uid="fresh_u1"),
                    "generated_text": "A",
                    "first_generated_token_text": "A",
                    "parsed_choice": "A",
                    "parser_status": "resolved",
                    "parser_signals": {"decision": "resolved_letter_first_token"},
                    "is_correct": True,
                    "layerwise": [
                        {
                            "layer_index": 0,
                            "candidate_logits": {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
                            "candidate_probs": {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1},
                            "candidate_entropy": 0.8,
                            "top_candidate": "A",
                            "top2_margin_prob": 0.6,
                            "projected_hidden_128": [0.0] * 128,
                        }
                    ],
                }
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(new_row, sort_keys=True) + "\n")
                return {
                    "pass": True,
                    "stopped_early": False,
                    "rows_written": 1,
                    "output_path": str(out_path),
                    "rows_per_second": 5.0,
                    "batch_sizes_used": [4],
                }

            with patch.object(mod, "run_baseline_for_model", side_effect=_fake_run_baseline_for_model), patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT_00A),
                    "--run-id",
                    run_id,
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                    "--max-prompts",
                    "1",
                ],
            ):
                rc = mod.main()

            self.assertEqual(rc, 0)
            rows = list(iter_jsonl(out_path))
            self.assertEqual(len(rows), 1)
            self.assertEqual(str(rows[0].get("prompt_uid")), "fresh_u1")
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage00a_fails_closed_on_empty_existing_manifest(self) -> None:
        mod = _load_stage00a_module()
        run_id = "test_v2_stage00a_empty_manifest_fail_closed"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)
        try:
            cfg_path = run_root / "cfg.yaml"
            (run_root / "manifests").mkdir(parents=True, exist_ok=True)
            (run_root / "manifests" / "ccc_baseline.jsonl").write_text("", encoding="utf-8")
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "data_scope:",
                        "  baseline_manifest_source: /Users/shaileshrana/shape-of-wisdom/prompt_packs/ccc_baseline_v1_3000.jsonl",
                        "  baseline_manifest_sha256: bfe2557316cb1e0eae6a684eb8de84885f74e446ac72f1847a7da80baf2de56c",
                        "  baseline_manifest_expected_rows_full: 3000",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT_00A),
                    "--run-id",
                    run_id,
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                ],
            ):
                rc = mod.main()

            self.assertEqual(rc, 2)
            report = json.loads((run_root / "v2" / "00a_generate_baseline_outputs.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertIn("manifest_nonempty", set(report.get("failing_gates") or []))
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage00a_fails_when_observed_rows_do_not_match_target_rows(self) -> None:
        mod = _load_stage00a_module()
        run_id = "test_v2_stage00a_rows_incomplete"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)
        try:
            cfg_path = run_root / "cfg.yaml"
            run_root.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "data_scope:",
                        "  baseline_manifest_source: /Users/shaileshrana/shape-of-wisdom/prompt_packs/ccc_baseline_v1_3000.jsonl",
                        "  baseline_manifest_expected_rows_full: 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            def _fake_run_baseline_for_model(*, run_id, model, out_path, **kwargs):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text("", encoding="utf-8")
                return {
                    "pass": True,
                    "stopped_early": False,
                    "rows_written": 0,
                    "output_path": str(out_path),
                    "rows_per_second": 3.0,
                    "batch_sizes_used": [1],
                }

            with patch.object(mod, "run_baseline_for_model", side_effect=_fake_run_baseline_for_model), patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT_00A),
                    "--run-id",
                    run_id,
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                    "--max-prompts",
                    "1",
                ],
            ):
                rc = mod.main()

            self.assertEqual(rc, 2)
            report = json.loads((run_root / "v2" / "00a_generate_baseline_outputs.report.json").read_text(encoding="utf-8"))
            self.assertIn("rows_complete_per_model", set(report.get("failing_gates") or []))
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage00a_full_mode_fails_when_expected_rows_contract_missing(self) -> None:
        mod = _load_stage00a_module()
        run_id = "test_v2_stage00a_expected_rows_missing"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)
        try:
            cfg_path = run_root / "cfg.yaml"
            manifest_path = run_root / "manifests" / "ccc_baseline.jsonl"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "prompt_uid": "u0",
                        "example_id": "e0",
                        "wrapper_id": "plain_exam",
                        "coarse_domain": "biology",
                        "prompt_text": "Question?\nA) yes\nB) no\nC) maybe\nD) none",
                        "correct_key": "A",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "data_scope:",
                        "  baseline_manifest_source: /Users/shaileshrana/shape-of-wisdom/prompt_packs/ccc_baseline_v1_3000.jsonl",
                        "  baseline_manifest_expected_rows_full: 0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            def _fake_run_baseline_for_model(*, run_id, model, out_path, **kwargs):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "model_id": str(model["model_id"]),
                            "model_revision": str(model["revision"]),
                            "prompt_uid": "u0",
                            "example_id": "e0",
                            "wrapper_id": "plain_exam",
                            "coarse_domain": "biology",
                            "resume_key": resume_key_for(model_id=str(model["model_id"]), prompt_uid="u0"),
                            "generated_text": "A",
                            "first_generated_token_text": "A",
                            "parsed_choice": "A",
                            "parser_status": "resolved",
                            "parser_signals": {"decision": "resolved_letter_first_token"},
                            "is_correct": True,
                            "layerwise": [
                                {
                                    "layer_index": 0,
                                    "candidate_logits": {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
                                    "candidate_probs": {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1},
                                    "candidate_entropy": 0.8,
                                    "top_candidate": "A",
                                    "top2_margin_prob": 0.6,
                                    "projected_hidden_128": [0.0] * 128,
                                }
                            ],
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return {
                    "pass": True,
                    "stopped_early": False,
                    "rows_written": 1,
                    "output_path": str(out_path),
                    "rows_per_second": 3.0,
                    "batch_sizes_used": [1],
                }

            with patch.object(mod, "run_baseline_for_model", side_effect=_fake_run_baseline_for_model), patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT_00A),
                    "--run-id",
                    run_id,
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                ],
            ):
                rc = mod.main()

            self.assertEqual(rc, 2)
            report = json.loads((run_root / "v2" / "00a_generate_baseline_outputs.report.json").read_text(encoding="utf-8"))
            self.assertIn("manifest_row_count_expected", set(report.get("failing_gates") or []))
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage00a_full_mode_fails_when_manifest_rows_mismatch_positive_expected(self) -> None:
        mod = _load_stage00a_module()
        run_id = "test_v2_stage00a_expected_rows_positive_mismatch"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)
        try:
            cfg_path = run_root / "cfg.yaml"
            manifest_path = run_root / "manifests" / "ccc_baseline.jsonl"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "prompt_uid": "u0",
                        "example_id": "e0",
                        "wrapper_id": "plain_exam",
                        "coarse_domain": "biology",
                        "prompt_text": "Question?\nA) yes\nB) no\nC) maybe\nD) none",
                        "correct_key": "A",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "data_scope:",
                        "  baseline_manifest_source: /Users/shaileshrana/shape-of-wisdom/prompt_packs/ccc_baseline_v1_3000.jsonl",
                        "  baseline_manifest_expected_rows_full: 2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            def _fake_run_baseline_for_model(*, run_id, model, out_path, **kwargs):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "model_id": str(model["model_id"]),
                            "model_revision": str(model["revision"]),
                            "prompt_uid": "u0",
                            "example_id": "e0",
                            "wrapper_id": "plain_exam",
                            "coarse_domain": "biology",
                            "resume_key": resume_key_for(model_id=str(model["model_id"]), prompt_uid="u0"),
                            "generated_text": "A",
                            "first_generated_token_text": "A",
                            "parsed_choice": "A",
                            "parser_status": "resolved",
                            "parser_signals": {"decision": "resolved_letter_first_token"},
                            "is_correct": True,
                            "layerwise": [
                                {
                                    "layer_index": 0,
                                    "candidate_logits": {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
                                    "candidate_probs": {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1},
                                    "candidate_entropy": 0.8,
                                    "top_candidate": "A",
                                    "top2_margin_prob": 0.6,
                                    "projected_hidden_128": [0.0] * 128,
                                }
                            ],
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return {
                    "pass": True,
                    "stopped_early": False,
                    "rows_written": 1,
                    "output_path": str(out_path),
                    "rows_per_second": 3.0,
                    "batch_sizes_used": [1],
                }

            with patch.object(mod, "run_baseline_for_model", side_effect=_fake_run_baseline_for_model), patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT_00A),
                    "--run-id",
                    run_id,
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                ],
            ):
                rc = mod.main()

            self.assertEqual(rc, 2)
            report = json.loads((run_root / "v2" / "00a_generate_baseline_outputs.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertIn("manifest_row_count_expected", set(report.get("failing_gates") or []))
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
