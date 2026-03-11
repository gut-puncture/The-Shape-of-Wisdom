from __future__ import annotations

from pathlib import Path

CHOICES: tuple[str, ...] = ("A", "B", "C", "D")
CHOICE_TO_INDEX = {choice: idx for idx, choice in enumerate(CHOICES)}

REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = REPO_ROOT / "analysis" / "no_new_inference_decision_dynamics"

DEFAULT_LAYERWISE = REPO_ROOT / "results" / "parquet" / "layerwise.parquet"
DEFAULT_DECISION_METRICS = REPO_ROOT / "results" / "parquet" / "decision_metrics.parquet"
DEFAULT_MANIFEST = REPO_ROOT / "prompt_packs" / "ccc_baseline_v1_3000.jsonl"
DEFAULT_OLD_CORE = REPO_ROOT / "paper" / "part1" / "data" / "part1_core.parquet"

MODEL_SHORT = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B-v0.3",
}

MODEL_LAYERS = {
    "Qwen/Qwen2.5-7B-Instruct": 28,
    "meta-llama/Llama-3.1-8B-Instruct": 32,
    "mistralai/Mistral-7B-Instruct-v0.3": 32,
}

MODEL_ORDER = tuple(MODEL_LAYERS.keys())
SEED = 12345
BOOTSTRAP_REPS = 1000

FUTURE_FLIP_Z_BINS = 10
FUTURE_FLIP_P_BINS = 20
SUMMARY_Z_BINS = 20
ALPHAS: tuple[float, ...] = (0.05, 0.10)
ANGULAR_Q_THRESHOLD = 1e-6
THETA_NAN_THRESHOLD = 1e-12
OLD_THRESHOLD_M = 0.75

BOUNDARY_WIDTHS_PROB = (0.05, 0.10, 0.20)
BOUNDARY_WIDTHS_MARGIN = (0.25, 0.50, 1.00)

COLOR_FINAL_CORRECT = "#0A6F69"
COLOR_FINAL_WRONG = "#C46A2D"
COLOR_BOUNDARY = "#7B6D8D"
COLOR_NEUTRAL = "#444444"
COLOR_MODEL = {
    "Qwen/Qwen2.5-7B-Instruct": "#1f4f7a",
    "meta-llama/Llama-3.1-8B-Instruct": "#a45418",
    "mistralai/Mistral-7B-Instruct-v0.3": "#3f7f3f",
}
DEPTH_CMAP = "cividis"

MAIN_FIGURES = (
    "fig01_state_decomposition",
    "fig02_depthwise_dynamics",
    "fig03_reversibility_map",
    "fig04_boundary_dwellers",
    "fig05_exemplar_trajectories",
)

APPENDIX_FIGURES = (
    "appendix_baseline_margin_smoothing",
    "appendix_baseline_switch_commitment",
    "appendix_per_model_reversibility_maps",
    "appendix_boundary_width_robustness",
    "appendix_travel_metric_distributions",
    "appendix_old_vs_empirical_commitment",
    "appendix_competitor_dynamics",
)

REQUIRED_MARKDOWN = (
    "README.md",
    "repo_inventory.md",
    "analysis_report.md",
    "figure_manifest.md",
    "paper_outline.md",
    "methods_for_general_cs_reader.md",
    "claim_discipline.md",
)

REQUIRED_PARQUETS = (
    "derived_data/layerwise_scores_canonical.parquet",
    "derived_data/layerwise_scores_with_state_metrics.parquet",
    "derived_data/trajectory_metrics.parquet",
    "derived_data/layerwise_future_flip_metrics.parquet",
)

REQUIRED_TABLES = (
    "tables/table_data_validation.csv",
    "tables/table_baseline_reproduction.csv",
    "tables/table_empirical_commitment_summary.csv",
    "tables/table_exemplar_prompts.csv",
    "tables/table_boundary_correlations.csv",
)

PERMUTATION_INDEX = {
    "A": (0, 1, 2, 3),
    "B": (1, 0, 2, 3),
    "C": (2, 0, 1, 3),
    "D": (3, 0, 1, 2),
}

INCORRECT_ORDER = {
    "A": ("B", "C", "D"),
    "B": ("A", "C", "D"),
    "C": ("A", "B", "D"),
    "D": ("A", "B", "C"),
}

REQUIRED_MANIFEST_COLUMNS = (
    "prompt_uid",
    "example_id",
    "correct_key",
    "subject",
    "coarse_domain",
    "wrapper_id",
    "dataset",
    "split",
    "question",
)

REQUIRED_LAYERWISE_COLUMNS = (
    "model_id",
    "model_revision",
    "prompt_uid",
    "example_id",
    "wrapper_id",
    "coarse_domain",
    "layer_index",
    "candidate_logits_json",
    "candidate_probs_json",
    "top_candidate",
)

REQUIRED_DECISION_COLUMNS = (
    "model_id",
    "prompt_uid",
    "layer_index",
    "correct_key",
    "competitor",
    "p_correct",
    "delta",
    "is_correct",
)

CANONICAL_COLUMNS = (
    "model_id",
    "model_revision",
    "model_name",
    "model_short",
    "prompt_uid",
    "example_id",
    "subject",
    "coarse_domain",
    "wrapper_id",
    "dataset",
    "split",
    "correct_option",
    "layer_index",
    "max_layer_index",
    "n_layers_logged",
    "z",
    "score_A",
    "score_B",
    "score_C",
    "score_D",
    "prob_A",
    "prob_B",
    "prob_C",
    "prob_D",
    "top_candidate",
    "top_tie_count",
    "top_is_tied",
    "competitor_tie_count",
    "competitor_is_tied",
    "final_predicted_option",
    "final_argmax_tie",
    "final_correct",
)

