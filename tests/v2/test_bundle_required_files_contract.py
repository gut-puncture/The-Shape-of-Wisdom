import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT11 = REPO_ROOT / "scripts" / "v2" / "11_generate_paper_assets.py"


def _load_script11():
    if str(SCRIPT11.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT11.parent))
    spec = importlib.util.spec_from_file_location("s11_bundle_contract", SCRIPT11)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT11}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBundleRequiredFilesContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod11 = _load_script11()

    def test_required_files_include_full_repro_contract(self) -> None:
        req = set(self.mod11.REQUIRED_FILES)
        expected = {
            "layerwise.parquet",
            "decision_metrics.parquet",
            "prompt_types.parquet",
            "spans.jsonl",
            "span_effects.parquet",
            "span_labels.parquet",
            "tracing_scalars.parquet",
            "attention_mass_by_span.parquet",
            "attention_contrib_by_span.parquet",
            "ablation_results.parquet",
            "patching_results.parquet",
            "span_deletion_causal.parquet",
            "negative_controls.parquet",
            "05_span_counterfactuals.report.json",
            "06_select_tracing_subset.report.json",
            "07_run_tracing.report.json",
            "08_attention_and_mlp_decomposition.report.json",
            "09_causal_tests.report.json",
            "10_causal_validation_tools.report.json",
        }
        missing = sorted(expected - req)
        self.assertFalse(missing, msg=f"bundle is missing required reproducibility artifacts: {missing}")

    def test_required_metadata_includes_preregistration_contract(self) -> None:
        req_meta = {Path(p).name for p in self.mod11.REQUIRED_METADATA_FILES}
        expected_meta = {
            "experiment_v2.yaml",
            "PAPER_OBJECTIVE_V3.md",
            "IMPLEMENTATION_PLAN_V3.md",
            "MODEL_NUANCES_V2.md",
            "PREREGISTERED_HYPOTHESES_V3.md",
        }
        missing = sorted(expected_meta - req_meta)
        self.assertFalse(missing, msg=f"bundle metadata contract missing required docs: {missing}")


if __name__ == "__main__":
    unittest.main()
