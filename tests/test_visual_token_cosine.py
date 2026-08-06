from pathlib import Path

import torch

from prismatic.conf.vla import Exp_JEPAVLA_Qwen25_VJEPA_0_5B_LIBERO_90
from prismatic.models.action_heads import VisualTokenCosineHead
from prismatic.models.vlms.prismatic import PrismaticVLM


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


def test_action_memory_uses_last_non_padding_tokens_per_sample() -> None:
    hidden = torch.arange(20, dtype=torch.float32).reshape(2, 10, 1)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    selected = PrismaticVLM._select_action_memory(hidden, attention_mask, num_action_tokens=3)

    torch.testing.assert_close(selected[0, :, 0], torch.tensor([7.0, 8.0, 9.0]))
    torch.testing.assert_close(selected[1, :, 0], torch.tensor([14.0, 15.0, 16.0]))


def test_public_recipe_keeps_single_visual_cosine_architecture() -> None:
    cfg = Exp_JEPAVLA_Qwen25_VJEPA_0_5B_LIBERO_90()
    script = (REPO_ROOT / "vla-scripts" / "run_visual_cosine_primary.sh").read_text()
    architecture_source = "\n".join(
        path.read_text()
        for path in (
            REPO_ROOT / "prismatic" / "conf" / "vla.py",
            REPO_ROOT / "prismatic" / "models" / "action_heads.py",
            REPO_ROOT / "prismatic" / "models" / "flow_gr00t_action_head.py",
            REPO_ROOT / "prismatic" / "models" / "vlms" / "prismatic.py",
            REPO_ROOT / "prismatic" / "vla" / "datasets" / "datasets.py",
            REPO_ROOT / "prismatic" / "vla" / "materialize.py",
            REPO_ROOT / "prismatic" / "training" / "train.py",
        )
    )

    assert 'RUN_ID_NOTE="visual-cosine-projector-allviews"' in script
    assert 'NPROC_PER_NODE=8' in script
    assert '--nproc-per-node "${NPROC_PER_NODE}"' in script
    assert "--module prismatic.training.train" in script
    assert cfg.expected_world_size == 8
    assert cfg.global_batch_size == 256
    assert cfg.per_device_batch_size == 32
    assert cfg.max_steps == 40_000

    forbidden = (
        "llm_prefix_bidirectional_attention",
        "visual_token_cosine_use_projector_target",
        "visual_token_cosine_layer_idx",
        "visual_token_cosine_projection_type",
        "visual_token_cosine_target_future_only",
        "action_queries",
        "aux_head",
        "vla-lora-last-n-train",
        "vla-vlm-peft-train",
        "DiT-B",
    )
    for option in forbidden:
        assert option not in script
        assert option not in architecture_source


def test_public_vla_config_matches_released_model() -> None:
    cfg = Exp_JEPAVLA_Qwen25_VJEPA_0_5B_LIBERO_90()

    assert (cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout) == (32, 64, 0.1)
    assert cfg.flow_gr00t_placeholder_tokens > 0
    assert cfg.lambda_visual_token_cosine == 0.5
    assert cfg.visual_token_pair_offset == 31
    assert cfg.d_action == 7
    assert cfg.d_proprio == 8


def test_removed_experimental_packages_are_not_published() -> None:
    removed_paths = (
        "jepa_wam",
        "lerobot",
        "meta",
        "pretrained_models",
        "experiments/robot/aloha",
        "experiments/robot/server_deploy",
        "prismatic/extern",
        "prismatic/preprocessing",
        "prismatic/models/flow_gr00t_jepa_action_head.py",
        "prismatic/training/train_utils.py",
        "prismatic/util/batching_utils.py",
    )
    for relative_path in removed_paths:
        path = REPO_ROOT / relative_path
        if path.is_dir():
            assert not any(path.rglob("*.py"))
        else:
            assert not path.exists()


def test_vla_scripts_only_expose_training_and_evaluation_launchers() -> None:
    public_scripts = sorted(path.name for path in (REPO_ROOT / "vla-scripts").glob("*.sh"))
    assert public_scripts == ["libero_plus.sh", "run_visual_cosine_primary.sh"]
    assert not (REPO_ROOT / "vla-scripts" / "train.py").exists()


def test_libero_plus_launcher_matches_native_checkpoint_evaluator() -> None:
    launcher = (REPO_ROOT / "vla-scripts" / "libero_plus.sh").read_text()
    evaluator = (REPO_ROOT / "experiments" / "robot" / "libero" / "run_libero_eval.py").read_text()

    for option in (
        "--pretrained_checkpoint",
        "--base_vlm",
        "--llm_checkpoint_path",
        "--vjepa_checkpoint_path",
        "--task_suite_name",
        "--libero_plus_categories",
        "--num_trials_per_task",
        "--save_rollouts",
    ):
        assert option in launcher
        assert option.removeprefix("--") in evaluator

    for removed_option in (
        "--model_family",
        "--action_head_type",
        "--num_images_in_input",
        "--rotation_representation",
        "--use_aux_head",
        "--use_minivlm",
        "--use_wandb",
    ):
        assert removed_option not in launcher


def test_public_launchers_require_explicit_asset_paths() -> None:
    training = (REPO_ROOT / "vla-scripts" / "run_visual_cosine_primary.sh").read_text()
    evaluation = (REPO_ROOT / "vla-scripts" / "libero_plus.sh").read_text()

    for variable in ("LIBERO_DATA", "QWEN_PATH", "VJEPA_CKPT", "BASE_VLM_RUN"):
        assert f'${{{variable}:?' in training
    for variable in ("QWEN_PATH", "VJEPA_CKPT", "BASE_VLM_RUN", "LIBERO_PATH"):
        assert f'${{{variable}:?' in evaluation

    assert "DEFAULT_CHECKPOINT" not in evaluation
    assert "JEPA_ENV" not in training
    assert "JEPA_ENV" not in evaluation
