from __future__ import annotations

from model_registry import load_model_specs


def test_model_registry_loads_system_prompt_handling(tmp_path) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "\n".join(
            [
                "models:",
                "  - name: demo",
                "    family: demo_family",
                "    size: 1b",
                "    model_id: demo/model",
                "    model_revision: rev1",
                "    tokenizer_revision: rev1",
                "    prompt_mode: chat",
                "    enabled: true",
                "    provisional: false",
                "    trust_remote_code: false",
                "    family_ready: false",
                "    local_files_only: false",
                "    hidden_state_projection_mode: apply_final_norm_except_last",
                "    system_prompt_handling: user_prefix",
            ]
        ),
        encoding="utf-8",
    )
    specs = load_model_specs(config_path)
    assert len(specs) == 1
    assert specs[0].system_prompt_handling == "user_prefix"


def test_model_registry_defaults_system_prompt_handling_to_system_role(tmp_path) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "\n".join(
            [
                "models:",
                "  - name: demo",
                "    family: demo_family",
                "    size: 1b",
                "    model_id: demo/model",
                "    model_revision: rev1",
                "    tokenizer_revision: rev1",
                "    prompt_mode: chat",
                "    enabled: true",
                "    provisional: false",
                "    trust_remote_code: false",
                "    family_ready: false",
                "    local_files_only: false",
                "    hidden_state_projection_mode: apply_final_norm_except_last",
            ]
        ),
        encoding="utf-8",
    )
    specs = load_model_specs(config_path)
    assert len(specs) == 1
    assert specs[0].system_prompt_handling == "system_role"
