# JEPA-WAM Public Architecture Specification

This document defines the architecture used by `vla-scripts/run_visual_cosine_primary.sh`. Experimental alternatives are intentionally outside this contract.

## 1. Inputs

Each LIBERO training example contains:

| Input | Shape | Description |
|---|---|---|
| current images | `[B, V, 3, H, W]` | primary and wrist views at time `t` |
| paired clip | `[B, V, 2, 3, H, W]` | each view at `[t, t + 31]` |
| text tokens | `[B, L]` | language instruction |
| proprio | `[B, 8]` | LIBERO robot state |
| actions | `[B, 8, 7]` | eight-step action chunk |

The public recipe uses two views (`V = 2`): primary and wrist.

## 2. Current-Image Encoding

The frozen V-JEPA 2.1 ViT-L encoder converts each current view into patch tokens:

```text
current images -> V-JEPA -> current_vjepa
current_vjepa: [B, V * N, D_jepa]
```

V-JEPA parameters are frozen. The pretrained VLM projector maps these tokens to the Qwen hidden dimension:

```text
projected_visual = vision_projector(current_vjepa)
projected_visual: [B, V * N, D_llm]
```

## 3. Qwen Sequence and Attention

The multimodal sequence is formed as:

```text
[BOS] [visual tokens] [text tokens] [action placeholder tokens]
```

Qwen2.5-0.5B processes this sequence with its native causal attention. No prefix-bidirectional mask or custom vision-text attention mask is constructed.

Consequences of the causal ordering:

- text tokens can attend to the preceding visual tokens;
- action placeholders can attend to visual and text tokens;
- a visual token only attends to earlier sequence positions, following normal Qwen behavior.

The final Qwen layer produces:

```text
qwen_hidden: [B, S, D_llm]
visual_hidden = qwen_hidden[:, 1 : 1 + V * N]
action_memory = qwen_hidden[:, -N_action_placeholders:]
```

## 4. Flow-Matching Action Head

The GR00T-style flow-matching action head is conditioned on:

- `action_memory`, taken only from the action-placeholder Qwen states;
- the current proprio state;
- noisy action tokens and the flow timestep.

The full Qwen hidden sequence is not supplied directly to the action head. Visual tokens are not separately prepended to `action_memory`.

The action loss is the flow-matching regression loss:

```text
L_action = mean(masked_flow_matching_error)
```

The action transformer uses its standard non-causal token interaction path.

## 5. Paired V-JEPA Target

For each view, the current image and its offset-31 image form a two-frame clip. The primary and wrist clips are encoded by the same frozen V-JEPA encoder:

```text
paired [current, offset-31] clips -> frozen V-JEPA -> pair_vjepa_target
pair_vjepa_target: [B, V, 1, H_p, W_p, D_jepa]
```

The temporal token dimension must be one. The target is flattened over view and spatial dimensions:

```text
target = reshape(pair_vjepa_target, [B, V * H_p * W_p, D_jepa])
target = stop_gradient(target)
```

The target is not passed through the pretrained VLM vision projector.

## 6. Visual-Token Cosine Head

The final-layer Qwen visual states are projected into the V-JEPA embedding space by a fixed two-layer MLP:

```python
pred = Linear(D_llm, 2 * D_jepa)(visual_hidden)
pred = GELU(pred)
pred = Linear(2 * D_jepa, D_jepa)(pred)
```

There is no convolutional projection alternative and no selectable intermediate Qwen layer.

Both prediction and target are L2-normalized along the embedding dimension. The loss is:

```text
pred_n   = normalize(pred, dim=-1)
target_n = normalize(target, dim=-1)

L_visual = mean(1 - sum(pred_n * target_n, dim=-1))
```

Gradients flow through the cosine MLP and the trainable Qwen LoRA parameters. They do not flow into the paired V-JEPA target.

## 7. Total Objective

The public model uses two losses:

```text
L_total = L_action + 0.5 * L_visual
```

The following objectives are disabled in this architecture:

- auxiliary future-V-JEPA decoder loss;
- joint JEPA action-head loss;
- language-model cross-entropy loss.

## 8. Fixed Training Configuration

| Component | Fixed value |
|---|---|
| vision encoder | V-JEPA 2.1 ViT-L, frozen |
| language model | Qwen2.5-0.5B with LoRA |
| attention | native Qwen causal attention |
| views | primary + wrist |
| action head | Flow GR00T |
| action conditioning | Qwen action-placeholder states |
| visual alignment head | `Linear -> GELU -> Linear` |
| visual target | raw detached paired V-JEPA tokens |
| pair offset | 31 |
| visual cosine weight | 0.5 |
| future observation window | 0 |
| rotation representation | axis-angle |

This table is the compatibility contract for public checkpoints produced by the supported training script.
