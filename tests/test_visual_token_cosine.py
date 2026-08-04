from pathlib import Path

import torch

from prismatic.conf.vla import Exp_JEPAVLA_Qwen25_VJEPA_0_5B_LIBERO_90
from prismatic.models.action_heads import VisualTokenCosineHead


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_visual_token_cosine_uses_trainable_mlp_and_detached_vjepa_target() -> None:
    torch.manual_seed(7)
    head = VisualTokenCosineHead(d_llm=8, d_target=12)
    llm_visual_tokens = torch.randn(2, 6, 8, requires_grad=True)
    vjepa_target = torch.randn(2, 6, 12, requires_grad=True)

    loss, projected = head(llm_visual_tokens, vjepa_target)
    loss.backward()

    assert projected.shape == (2, 6, 12)
    assert llm_visual_tokens.grad is not None
    assert head.fc1.weight.grad is not None
    assert head.fc2.weight.grad is not None
    assert vjepa_target.grad is None


def test_visual_token_cosine_is_zero_for_identical_normalized_embeddings() -> None:
    torch.manual_seed(11)
    head = VisualTokenCosineHead(d_llm=8, d_target=12)
    llm_visual_tokens = torch.randn(2, 6, 8)
    matching_target = head.align_dimension(llm_visual_tokens).detach()

    loss, _ = head(llm_visual_tokens, matching_target)

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0.0)


def test_public_recipe_keeps_single_visual_cosine_architecture() -> None:
    script = (REPO_ROOT / "vla-scripts" / "run_visual_cosine_primary.sh").read_text()
    architecture_source = "\n".join(
        path.read_text()
        for path in (
            REPO_ROOT / "prismatic" / "conf" / "vla.py",
            REPO_ROOT / "prismatic" / "models" / "action_heads.py",
            REPO_ROOT / "prismatic" / "models" / "vlms" / "prismatic.py",
            REPO_ROOT / "prismatic" / "vla" / "datasets" / "datasets.py",
            REPO_ROOT / "prismatic" / "vla" / "materialize.py",
            REPO_ROOT / "vla-scripts" / "train.py",
        )
    )

    assert "--run_id_note visual-cosine-projector-allviews" in script
    assert "--vla.expected_world_size 8" in script
    assert "--vla.global_batch_size 256" in script
    assert "--vla.max_steps 40000" in script

    forbidden = (
        "llm_prefix_bidirectional_attention",
        "visual_token_cosine_use_projector_target",
        "visual_token_cosine_layer_idx",
        "visual_token_cosine_projection_type",
        "visual_token_cosine_target_future_only",
    )
    for option in forbidden:
        assert option not in script
        assert option not in architecture_source


def test_public_vla_config_matches_released_model() -> None:
    cfg = Exp_JEPAVLA_Qwen25_VJEPA_0_5B_LIBERO_90()

    assert cfg.image_sequence_len == 2
    assert cfg.use_wrist_image is True
    assert cfg.freeze_vision_backbone is True
    assert cfg.use_lora is True
    assert (cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout) == (32, 64, 0.1)
    assert cfg.action_head_type == "flow_gr00t"
    assert cfg.flow_gr00t_use_full_llm_hidden is False
    assert cfg.use_aux_head is False
    assert cfg.use_visual_token_cosine_head is True
    assert cfg.lambda_visual_token_cosine == 0.5
    assert cfg.visual_token_pair_offset == 31
    assert cfg.future_obs_window_size == 0
    assert cfg.rotation_representation == "axis_angle"
