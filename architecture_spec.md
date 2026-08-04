# JEPA-VLA 架构规格

本文档描述一个基于 V-JEPA 2 + 轻量 LLM + Flow Matching 的 VLA 模型架构,整体骨架参考 VLA-Adapter / Prismatic-VLM,主要改动:

1. 视觉 backbone 从 DINOv2+SigLIP 替换为 **V-JEPA 2** (frozen encoder only)
2. 加入 **Aux Head**,监督 LLM 中间表征,使其能预测未来帧的 V-JEPA embedding
3. Action Head 使用 **GR00T 风格的 16 层交替 self/cross-attention + flow matching**

实现目标:可训练的 JEPA-VLA 模型,端到端训练,主 loss 为 action flow matching loss,辅助 loss 为未来帧 JEPA embedding 预测的 normalized MSE。

---

## 0. Config 约定

以下值来自训练配置文件(config),不在此写死:

| 字段 | 含义 | 典型值 |
|---|---|---|
| `num_views` | 视角数 | 2 或 3 |
| `num_views_max` | 视角数上限(用于 position embedding 预留) | 3 |
| `proprio_dim` | 本体感知维度 | 取决于机器人,如 14(双臂)、7(单臂) |
| `action_dim` | 动作维度 | 取决于机器人,如 14 或 7 |
| `action_horizon` | 一次预测的动作步数 | 8 |
| `llm_name` | LLM 选择 | `Qwen2.5-0.5B` |
| `llm_hidden_dim` | LLM hidden dim | 896 (Qwen2.5-0.5B) |
| `vjepa_hidden_dim` | V-JEPA 2 encoder 输出维度 | 1024 (ViT-L) |

本文档里写 `D_llm`、`D_jepa`、`V`、`H_a`、`D_proprio`、`D_action` 等符号,对应上面的 config 字段。

---

## 1. 总体数据流

```
输入:
  images           [B, V, 3, 224, 224]        当前帧,多视角
  text_tokens      [B, L_text]                语言指令 token ids
  proprio          [B, D_proprio]             本体感知
  future_frames    [B, V, 8, 3, 224, 224]     未来 8 帧,训练时提供

  ┌─────────────────────────────────────────────────────────┐
  │ 1. V-JEPA encoder (frozen)                               │
  │    images ─ 复制帧 ─> [B*V, 2, 3, 224, 224]             │
  │           ─> [B, V, 196, 1024]                           │
  └─────────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 2. Vision Projector (2 层 MLP + 视角 PE)                  │
  │    [B, V, 196, 1024] ─> [B, V*196, D_llm]                │
  └─────────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 3. 语言 embedding (LLM 原生 embed_tokens)                 │
  │    text_tokens ─> [B, L_text, D_llm]                     │
  └─────────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 4. 序列拼接(视觉在前,语言在后)                            │
  │    concat ─> [B, V*196 + L_text, D_llm]                  │
  └─────────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 5. LLM (Qwen2.5-0.5B, 冻结 backbone + LoRA)              │
  │    输出 last layer hidden state: [B, L, D_llm]            │
  └─────────────────────────────────────────────────────────┘
             │                          │
             ▼                          ▼
  ┌──────────────────────┐   ┌──────────────────────────┐
  │ 6. Action Head       │   │ 7. Aux Head              │
  │    (GR00T, 16 层)    │   │    (12 层 cross-decoder) │
  │                      │   │                          │
  │  条件: last hidden   │   │  memory: 整个 hidden seq │
  │  输入: proprio +     │   │  query:  learnable PE    │
  │        noisy action  │   │                          │
  │                      │   │                          │
  │  输出: [B, H_a, D_a] │   │  输出: [B, V, 4, 14, 14, │
  │       (flow matching)│   │        D_jepa]           │
  └──────────────────────┘   └──────────────────────────┘
             │                          │
             ▼                          ▼
      L_action                     L_aux (训练时)
```

---

## 2. 模块 1:V-JEPA Encoder (frozen)

### 2.1 输入处理

V-JEPA 2 要求时间维度 T≥2。当前帧只有一张图,通过**复制帧**绕过:

```python
# images: [B, V, 3, 224, 224]
x = images.unsqueeze(2).repeat(1, 1, 2, 1, 1, 1)  # [B, V, 2, 3, 224, 224]
x = x.reshape(B * V, 2, 3, 224, 224)
```

### 2.2 Encoder forward

```python
with torch.no_grad():
    # V-JEPA 2 ViT-L, 冻结, 使用官方预训练权重
    # patch=16, tubelet=2, 输出 hidden_dim=1024
    out = vjepa_encoder(x)  # [B*V, N_tokens, 1024]
    # N_tokens = (T/2) * (H/16) * (W/16) = 1 * 14 * 14 = 196

# reshape 回多视角
vision_emb = out.reshape(B, V, 196, 1024)  # [B, V, 196, 1024]
```

### 2.3 约束

- **encoder 全程冻结**(`requires_grad=False`, `no_grad()` 包裹)
- 使用官方 V-JEPA 2 或 V-JEPA 2.1 权重
- 不使用 V-JEPA 的 predictor(只用 encoder)

---

## 3. 模块 2:Vision Projector

把 V-JEPA 输出映射到 LLM 空间,加视角 position embedding。

### 3.1 结构

```python
class VisionProjector(nn.Module):
    """Prismatic 风格 2 层 MLP + view PE"""
    def __init__(self, d_jepa=1024, d_llm=896, num_views_max=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_jepa, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm),
        )
        self.view_pe = nn.Parameter(
            torch.randn(num_views_max, d_llm) * 0.02
        )

    def forward(self, vision_emb):
        # vision_emb: [B, V, 196, D_jepa]
        B, V, N, _ = vision_emb.shape
        x = self.proj(vision_emb)                   # [B, V, 196, D_llm]
        x = x + self.view_pe[:V, None, :]           # 加视角 PE(broadcast)
        x = x.reshape(B, V * N, -1)                 # [B, V*196, D_llm]
        return x
```

### 3.2 参数量

- MLP: `D_jepa × D_llm + D_llm × D_llm` ≈ 1.7M
- view_pe: `num_views_max × D_llm` ≈ 3k
- **总计 ≈ 1.7M**

---

## 4. 模块 3:语言 Embedding

直接复用 LLM 的 `embed_tokens`,不是独立模块:

```python
text_emb = llm.get_input_embeddings()(text_tokens)  # [B, L_text, D_llm]
```

---

## 5. 模块 4:序列拼接

**顺序:视觉在前,语言在后。**

这样 causal attention 下 language token 可以看到所有 visual token,其 hidden state 会融合视觉-语言信息。这对 aux head cross-attention 非常关键——语言 token 位置的 hidden state 会是信息最丰富的部分。

```python
llm_input = torch.cat([vision_tokens, text_emb], dim=1)
# [B, V*196 + L_text, D_llm]

# attention mask: 全 1(不需要 pad,因为序列内部没有 pad)
attention_mask = torch.ones(B, llm_input.shape[1], dtype=torch.long)
```

**序列长度估算**:
- V=2, L_text=20: L = 392 + 20 = 412
- V=3, L_text=20: L = 588 + 20 = 608

---

## 6. 模块 5:LLM

### 6.1 基础配置

- 模型: Qwen2.5-0.5B(可在 config 切换)
- Backbone: **冻结**(`requires_grad=False`)
- LoRA: 应用于所有 attention 投影(`q_proj`, `k_proj`, `v_proj`, `o_proj`),rank=32, alpha=64
- FFN 层不加 LoRA(节省参数,后续如需再加)

### 6.2 Forward

```python
llm_output = llm(
    inputs_embeds=llm_input,
    attention_mask=attention_mask,
    output_hidden_states=True,
    use_cache=False,
)
llm_hidden = llm_output.hidden_states[-1]  # [B, L, D_llm]
```

### 6.3 输出接口

Action head 和 Aux head 从 `llm_hidden` 取不同形式的信息:

```python
# 给 Action head: 取最后一个 token
z_action = llm_hidden[:, -1, :]     # [B, D_llm]

# 给 Aux head: 整个序列
aux_memory = llm_hidden             # [B, L, D_llm]
```

---

## 7. 模块 6:Action Head (GR00T 风格 Flow Matching)

### 7.1 总体结构

参考 GR00T N1 的 action expert 设计:16 层交替结构,**奇数层 self-attention,偶数层 cross-attention**。Cross-attention 的 K/V 来自 LLM 的 last hidden state(`z_action`)。

输入 token 序列由两部分组成:
1. **State token**:proprio 经过 state encoder 变成 1 个 token
2. **Noisy action tokens**:加噪后的 action,`H_a` 个 token(`H_a = action_horizon = 8`)

```
State token  [B, 1,   D_a]  ──┐
Noisy tokens [B, H_a, D_a]  ──┴─ concat ─> [B, 1+H_a, D_a]
                                              │
                                              ▼
                                  16 层 Transformer
                             (奇数 self-attn, 偶数 cross-attn)
                                              │
                                              ▼
                                 取后 H_a 个 token,
                                 投影到 D_action 维度
                                              │
                                              ▼
                              预测的 flow velocity [B, H_a, D_action]
```

### 7.2 子模块 A:State Encoder

把 proprio 编码为单个 token:

```python
class StateEncoder(nn.Module):
    def __init__(self, d_proprio, d_a):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_proprio, d_a),
            nn.SiLU(),
            nn.Linear(d_a, d_a),
        )

    def forward(self, proprio):
        # proprio: [B, D_proprio]
        x = self.encoder(proprio)       # [B, D_a]
        return x.unsqueeze(1)           # [B, 1, D_a]
```

### 7.3 子模块 B:Noisy Action Embedding

Flow matching 的输入是带噪 action。GR00T 使用 **Beta 分布采样时间步 t**,然后线性插值加噪:

```python
# 训练时采样 flow matching 时间步
# GR00T 用 Beta(1.5, 1.0) 偏向 t→0(噪声多的一端),学习更难的去噪
t = torch.distributions.Beta(1.5, 1.0).sample((B,))      # [B]
t = t.to(device)

# 线性插值加噪
noise = torch.randn_like(action_gt)                       # [B, H_a, D_action]
action_noisy = (1 - t[:, None, None]) * noise \
             + t[:, None, None] * action_gt               # [B, H_a, D_action]

# flow matching 的 target velocity
velocity_gt = action_gt - noise                           # [B, H_a, D_action]
```

将 noisy action 嵌入到 transformer 空间,加时间步条件:

```python
class NoisyActionEmbed(nn.Module):
    def __init__(self, d_action, d_a, horizon):
        super().__init__()
        self.action_proj = nn.Linear(d_action, d_a)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_a),
            nn.SiLU(),
            nn.Linear(d_a, d_a),
        )
        self.pos_embed = nn.Parameter(torch.randn(horizon, d_a) * 0.02)

    def forward(self, action_noisy, t):
        # action_noisy: [B, H_a, D_action]
        # t:            [B]
        x = self.action_proj(action_noisy)                # [B, H_a, D_a]
        t_emb = self.time_mlp(t.unsqueeze(-1))            # [B, D_a]
        x = x + t_emb.unsqueeze(1)                        # broadcast 加时间
        x = x + self.pos_embed.unsqueeze(0)               # 加 action step 位置
        return x                                          # [B, H_a, D_a]
```

### 7.4 子模块 C:16 层交替 Transformer

**奇数层(1, 3, 5, ...,共 8 层)= Self-Attention Block**:

```python
class SelfAttnBlock(nn.Module):
    def __init__(self, d_a, n_heads, ffn_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_a)
        self.attn  = nn.MultiheadAttention(d_a, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_a)
        self.ffn   = nn.Sequential(
            nn.Linear(d_a, d_a * ffn_ratio),
            nn.GELU(),
            nn.Linear(d_a * ffn_ratio, d_a),
        )

    def forward(self, x, cond=None):
        # cond 未使用(兼容统一 forward 接口)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x
```

**偶数层(2, 4, 6, ...,共 8 层)= Cross-Attention Block**:

```python
class CrossAttnBlock(nn.Module):
    def __init__(self, d_a, d_llm, n_heads, ffn_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_a)
        self.cond_proj = nn.Linear(d_llm, d_a)    # LLM hidden → action dim
        self.attn  = nn.MultiheadAttention(d_a, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_a)
        self.ffn   = nn.Sequential(
            nn.Linear(d_a, d_a * ffn_ratio),
            nn.GELU(),
            nn.Linear(d_a * ffn_ratio, d_a),
        )

    def forward(self, x, cond):
        # cond: [B, D_llm]  (last hidden state)
        h = self.norm1(x)
        k_v = self.cond_proj(cond).unsqueeze(1)   # [B, 1, D_a]
        h, _ = self.attn(h, k_v, k_v)             # cross-attn
        x = x + h
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x
```

**组装**:

```python
class ActionHeadBackbone(nn.Module):
    def __init__(self, d_a=1024, d_llm=896, n_heads=16, num_layers=16):
        super().__init__()
        blocks = []
        for i in range(num_layers):
            if i % 2 == 0:   # 0, 2, 4, ... (即"单数层"的 0-indexed)
                blocks.append(SelfAttnBlock(d_a, n_heads))
            else:
                blocks.append(CrossAttnBlock(d_a, d_llm, n_heads))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, cond):
        for block in self.blocks:
            x = block(x, cond)
        return x
```

> **注意**:用户描述"单数是 self,双数是 cross",这里按 0-indexed 实现时,第 0、2、4... 层为 self(对应人类编号的第 1、3、5... 层,即单数),第 1、3、5... 层为 cross(对应人类编号的双数)。如果 GR00T 原版实现相反,请根据官方代码对齐。

### 7.5 子模块 D:输出头

```python
class ActionOutputHead(nn.Module):
    def __init__(self, d_a, d_action):
        super().__init__()
        self.norm = nn.LayerNorm(d_a)
        self.proj = nn.Linear(d_a, d_action)

    def forward(self, x):
        # x: [B, 1 + H_a, D_a]
        x = x[:, 1:, :]              # 丢弃 state token,只留 action tokens
        x = self.norm(x)
        return self.proj(x)          # [B, H_a, D_action]
```

### 7.6 Action Head 整体 Forward(训练时)

```python
class ActionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.state_enc  = StateEncoder(cfg.D_proprio, cfg.d_a)
        self.noisy_emb  = NoisyActionEmbed(cfg.D_action, cfg.d_a, cfg.H_a)
        self.backbone   = ActionHeadBackbone(
            d_a=cfg.d_a, d_llm=cfg.D_llm,
            n_heads=cfg.n_heads_action, num_layers=16
        )
        self.out_head   = ActionOutputHead(cfg.d_a, cfg.D_action)

    def forward(self, z_action, proprio, action_gt):
        B = proprio.shape[0]

        # 1. 采样 flow matching 时间 + 加噪
        t = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(proprio.device)
        noise = torch.randn_like(action_gt)
        action_noisy = (1 - t[:, None, None]) * noise \
                     + t[:, None, None] * action_gt
        velocity_gt = action_gt - noise

        # 2. Token 化
        state_tok = self.state_enc(proprio)                  # [B, 1, D_a]
        act_tok   = self.noisy_emb(action_noisy, t)          # [B, H_a, D_a]
        x = torch.cat([state_tok, act_tok], dim=1)           # [B, 1+H_a, D_a]

        # 3. 过 backbone (cond 是 LLM last hidden)
        x = self.backbone(x, cond=z_action)                  # [B, 1+H_a, D_a]

        # 4. 输出 velocity 预测
        v_pred = self.out_head(x)                            # [B, H_a, D_action]

        # 5. Flow matching loss
        loss = F.mse_loss(v_pred, velocity_gt)
        return loss, v_pred
```

### 7.7 推理时的 Flow Matching Sampling

训练时采样单个 t 算 loss;推理时从 t=0 (纯噪声) 积分到 t=1 (action):

```python
@torch.no_grad()
def sample_action(action_head, z_action, proprio, num_steps=10):
    B = proprio.shape[0]
    device = proprio.device

    # 初始化为纯噪声
    x = torch.randn(B, H_a, D_action, device=device)

    # 欧拉积分
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((B,), step * dt, device=device)

        state_tok = action_head.state_enc(proprio)
        act_tok   = action_head.noisy_emb(x, t)
        tokens = torch.cat([state_tok, act_tok], dim=1)

        tokens = action_head.backbone(tokens, cond=z_action)
        v_pred = action_head.out_head(tokens)

        x = x + v_pred * dt

    return x   # [B, H_a, D_action]
```

### 7.8 Action Head 超参(config 字段)

| 字段 | 默认值 |
|---|---|
| `d_a` (action head internal dim) | 1024 |
| `n_heads_action` | 16 |
| `num_layers_action` | 16 |
| `ffn_ratio_action` | 4 |
| `flow_steps_inference` | 10 |
| `beta_alpha` / `beta_beta` | 1.5 / 1.0 |

### 7.9 参数量(d_a=1024, 16 层)

- State encoder:D_proprio × 1024 + 1024 × 1024 ≈ 1M(忽略 proprio_dim)
- Noisy action embed:D_action × 1024 + 时间 MLP + pos_embed ≈ 1.1M
- 每个 Self-attn block:4 × 1024² + 2 × 1024 × 4096 ≈ 12.6M
- 每个 Cross-attn block:(4 × 1024² + 1024 × 896) + FFN ≈ 13.5M
- 8 self + 8 cross ≈ 208M
- Output head:~1M
- **总计 ≈ 210M**

> **注意**:这个规模比之前估计的 40M 大很多。如果想控制规模,可以降 `d_a` 到 768(参数量降到 ~120M)或降层数。保持 16 层 d_a=1024 更接近 GR00T 原版。

---

## 8. 模块 7:Aux Head (12 层, ~110M)

### 8.1 总体结构

Cross-attention decoder,以可学习 query 为输出位置占位,attend 到 LLM 整个 hidden sequence,预测未来 JEPA embedding。

```
Learnable queries [B, V*T*H*W, D_aux]   (T=4, H=W=14)
                    │
                    ▼
         Input projections (query_proj, memory_proj)
                    │
                    ▼
       12 层 Decoder Block:
         [CrossAttn(memory) → SelfAttn → FFN]
                    │
                    ▼
       LayerNorm + Output projection (D_aux → D_jepa)
                    │
                    ▼
       Reshape → [B, V, 4, 14, 14, D_jepa]
```

### 8.2 超参

| 字段 | 值 |
|---|---|
| `d_aux` (internal dim) | 768 |
| `n_heads_aux` | 12 |
| `num_layers_aux` | **12** |
| `ffn_ratio_aux` | 4 |
| `aux_T` (时间 chunk 数) | 4 |
| `aux_H`, `aux_W` (空间网格) | 14, 14 |

### 8.3 子模块 A:Learnable Queries

```python
class AuxQueries(nn.Module):
    """通过视角/时间/空间 PE 广播相加构造 query"""
    def __init__(self, num_views_max=3, T=4, H=14, W=14, d=768):
        super().__init__()
        self.view_pe    = nn.Parameter(torch.randn(num_views_max, d) * 0.02)
        self.time_pe    = nn.Parameter(torch.randn(T, d) * 0.02)
        self.spatial_pe = nn.Parameter(torch.randn(H, W, d) * 0.02)
        self.T, self.H, self.W = T, H, W

    def forward(self, B, V):
        v = self.view_pe[:V][:, None, None, None, :]     # [V, 1, 1, 1, d]
        t = self.time_pe[None, :, None, None, :]         # [1, T, 1, 1, d]
        s = self.spatial_pe[None, None, :, :, :]         # [1, 1, H, W, d]
        q = v + t + s                                    # [V, T, H, W, d]
        q = q.reshape(1, V * self.T * self.H * self.W, -1)
        q = q.expand(B, -1, -1).contiguous()             # [B, N_q, d]
        return q
```

N_q 计算:
- V=2: N_q = 2 × 4 × 14 × 14 = 1568
- V=3: N_q = 2352

### 8.4 子模块 B:Input Projections

```python
self.query_proj  = nn.Linear(d_aux, d_aux)       # Q 侧自投影
self.memory_proj = nn.Linear(D_llm, d_aux)       # K/V 侧从 LLM 投影
```

### 8.5 子模块 C:Aux Decoder Block

标准 pre-norm transformer decoder block:

```python
class AuxDecoderBlock(nn.Module):
    def __init__(self, d, n_heads, ffn_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

        self.norm3 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * ffn_ratio),
            nn.GELU(),
            nn.Linear(d * ffn_ratio, d),
        )

    def forward(self, queries, memory):
        # Cross-attention: query → memory
        h = self.norm1(queries)
        h, _ = self.cross_attn(h, memory, memory)
        queries = queries + h

        # Self-attention: query 之间
        h = self.norm2(queries)
        h, _ = self.self_attn(h, h, h)
        queries = queries + h

        # FFN
        h = self.norm3(queries)
        queries = queries + self.ffn(h)

        return queries
```

### 8.6 Aux Head 整体 Forward

```python
class AuxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.queries = AuxQueries(
            num_views_max=cfg.num_views_max,
            T=cfg.aux_T, H=cfg.aux_H, W=cfg.aux_W,
            d=cfg.d_aux,
        )
        self.query_proj  = nn.Linear(cfg.d_aux, cfg.d_aux)
        self.memory_proj = nn.Linear(cfg.D_llm, cfg.d_aux)

        self.blocks = nn.ModuleList([
            AuxDecoderBlock(cfg.d_aux, cfg.n_heads_aux, cfg.ffn_ratio_aux)
            for _ in range(cfg.num_layers_aux)   # 12 层
        ])

        self.final_norm  = nn.LayerNorm(cfg.d_aux)
        self.output_proj = nn.Linear(cfg.d_aux, cfg.D_jepa)

        self.T, self.H, self.W = cfg.aux_T, cfg.aux_H, cfg.aux_W

    def forward(self, llm_hidden, V):
        # llm_hidden: [B, L, D_llm]
        # V: 当前 batch 的视角数(可能 < num_views_max)
        B = llm_hidden.shape[0]

        queries = self.queries(B, V)                     # [B, N_q, d_aux]
        queries = self.query_proj(queries)
        memory  = self.memory_proj(llm_hidden)           # [B, L, d_aux]

        for block in self.blocks:
            queries = block(queries, memory)

        queries = self.final_norm(queries)
        out = self.output_proj(queries)                  # [B, N_q, D_jepa]
        out = out.reshape(B, V, self.T, self.H, self.W, -1)
        return out
```

### 8.7 参数量(d=768, 12 层)

- Position embeddings:(3 + 4 + 196) × 768 ≈ 16 万
- query_proj:768² ≈ 60 万
- memory_proj:896 × 768 ≈ 69 万
- 每层 Decoder block:
  - Cross-attn:4 × 768² ≈ 2.36M
  - Self-attn:4 × 768² ≈ 2.36M
  - FFN:2 × 768 × 3072 ≈ 4.72M
  - 小计 ≈ 9.44M
- 12 层 ≈ 113M
- final_norm + output_proj:~80 万
- **总计 ≈ 115M**

---

## 9. 模块 8:Target 生成 & Aux Loss

### 9.1 Target

训练 batch 要提供未来 8 帧的 RGB,通过同一个 V-JEPA encoder 生成 target:

```python
def compute_aux_target(future_frames, vjepa_encoder):
    # future_frames: [B, V, 8, 3, 224, 224]
    B, V, T, C, H, W = future_frames.shape
    x = future_frames.reshape(B * V, T, C, H, W)

    with torch.no_grad():
        # V-JEPA encoder 吃 8 帧,tubelet=2 压成 4 时空 chunk
        out = vjepa_encoder(x)                       # [B*V, 4*14*14, D_jepa]
        target = out.reshape(B, V, 4, 14, 14, -1)    # [B, V, 4, 14, 14, D_jepa]

    return target.detach()
```

### 9.2 Loss

```python
def aux_loss(pred, target):
    # pred, target: [B, V, 4, 14, 14, D_jepa]
    # 两侧都做 per-patch LayerNorm(无可学习参数)
    pred_n   = F.layer_norm(pred,   [pred.shape[-1]])
    target_n = F.layer_norm(target, [target.shape[-1]])
    return F.mse_loss(pred_n, target_n)
```

---

## 10. 总 Loss 与训练调度

### 10.1 总 Loss

```python
loss = loss_action + lambda_aux * loss_aux
```

### 10.2 λ_aux 调度

- 前 10% 训练步数:`λ_aux = 1.0`(warm-up,先学视觉预测)
- 之后 cosine decay 到 `λ_aux = 0.2`

```python
def get_lambda_aux(step, total_steps, warmup_ratio=0.1,
                   lambda_init=1.0, lambda_final=0.2):
    warmup_steps = int(total_steps * warmup_ratio)
    if step < warmup_steps:
        return lambda_init
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return lambda_final + (lambda_init - lambda_final) * cosine
```

### 10.3 训练/推理差异

- **训练时**:计算 action loss + aux loss,aux head 参与前向。
- **推理时**:只运行 action head 做 flow matching sampling,aux head **不运行**(可以直接从模块列表里跳过以节省显存和时间)。

---

## 11. 可训练参数汇总

| 组件 | 参数量 | 训练? |
|---|---|---|
| V-JEPA 2 ViT-L encoder | ~300M | 冻结 |
| Vision Projector | ~1.7M | ✓ |
| Qwen2.5-0.5B 本体 | 500M | 冻结 |
| LoRA (rank 32) | ~5M | ✓ |
| Action Head (GR00T, 16 层, d_a=1024) | ~210M | ✓ |
| **Aux Head (12 层, d_aux=768)** | **~115M** | ✓ |
| **可训练参数总计** | **~332M** | |
| **全模型参数** | **~1.13B** | |

---

## 12. 主模型组装

```python
class JepaVLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 1. V-JEPA encoder (frozen)
        self.vjepa = load_vjepa2_encoder(cfg.vjepa_checkpoint)
        for p in self.vjepa.parameters():
            p.requires_grad = False

        # 2. Vision projector
        self.vision_proj = VisionProjector(
            d_jepa=cfg.D_jepa, d_llm=cfg.D_llm,
            num_views_max=cfg.num_views_max,
        )

        # 3. LLM + LoRA
        self.llm = load_llm_with_lora(cfg.llm_name, lora_rank=cfg.lora_rank)

        # 4. Action head (GR00T)
        self.action_head = ActionHead(cfg)

        # 5. Aux head
        self.aux_head = AuxHead(cfg)

    def encode_vision(self, images):
        """images: [B, V, 3, 224, 224] → [B, V, 196, D_jepa]"""
        B, V = images.shape[:2]
        x = images.unsqueeze(2).repeat(1, 1, 2, 1, 1, 1)
        x = x.reshape(B * V, 2, 3, 224, 224)
        with torch.no_grad():
            emb = self.vjepa(x)                              # [B*V, 196, D_jepa]
        return emb.reshape(B, V, 196, -1)

    def encode_future(self, future_frames):
        """future_frames: [B, V, 8, 3, 224, 224] → [B, V, 4, 14, 14, D_jepa]"""
        B, V, T = future_frames.shape[:3]
        x = future_frames.reshape(B * V, T, 3, 224, 224)
        with torch.no_grad():
            out = self.vjepa(x)                              # [B*V, 4*14*14, D_jepa]
        return out.reshape(B, V, 4, 14, 14, -1).detach()

    def forward(self, batch):
        images        = batch["images"]           # [B, V, 3, 224, 224]
        text_tokens   = batch["text_tokens"]      # [B, L_text]
        proprio       = batch["proprio"]          # [B, D_proprio]
        action_gt     = batch["action"]           # [B, H_a, D_action]
        future_frames = batch.get("future_frames")  # [B, V, 8, 3, 224, 224]
        V = images.shape[1]

        # 视觉
        vision_emb = self.encode_vision(images)                # [B, V, 196, D_jepa]
        vision_tok = self.vision_proj(vision_emb)              # [B, V*196, D_llm]

        # 语言
        text_emb = self.llm.get_input_embeddings()(text_tokens)  # [B, L_text, D_llm]

        # 拼接
        llm_input = torch.cat([vision_tok, text_emb], dim=1)

        # LLM
        llm_out = self.llm(
            inputs_embeds=llm_input,
            output_hidden_states=True,
            use_cache=False,
        )
        llm_hidden = llm_out.hidden_states[-1]                 # [B, L, D_llm]

        # Action head
        z_action = llm_hidden[:, -1, :]                        # [B, D_llm]
        loss_action, _ = self.action_head(z_action, proprio, action_gt)

        losses = {"action": loss_action}

        # Aux head (仅训练且有 future_frames 时)
        if self.training and future_frames is not None:
            aux_pred = self.aux_head(llm_hidden, V=V)
            aux_target = self.encode_future(future_frames)
            losses["aux"] = aux_loss(aux_pred, aux_target)

        return losses

    @torch.no_grad()
    def predict_action(self, batch, num_flow_steps=10):
        """推理时调用,只跑 action head"""
        images      = batch["images"]
        text_tokens = batch["text_tokens"]
        proprio     = batch["proprio"]

        vision_emb = self.encode_vision(images)
        vision_tok = self.vision_proj(vision_emb)
        text_emb   = self.llm.get_input_embeddings()(text_tokens)
        llm_input  = torch.cat([vision_tok, text_emb], dim=1)

        llm_out = self.llm(
            inputs_embeds=llm_input,
            output_hidden_states=True,
            use_cache=False,
        )
        llm_hidden = llm_out.hidden_states[-1]
        z_action = llm_hidden[:, -1, :]

        return sample_action(self.action_head, z_action, proprio,
                             num_steps=num_flow_steps)
```

---

## 13. Config 示例

```yaml
# config/default.yaml

# 维度
D_jepa:      1024        # V-JEPA 2 ViT-L
D_llm:       896         # Qwen2.5-0.5B
D_proprio:   14          # 双臂例
D_action:    14          # 双臂例
H_a:         8           # action_horizon
num_views:   2
num_views_max: 3

# V-JEPA
vjepa_checkpoint: "facebook/vjepa2-vitl-fpc64-256"
vjepa_num_patches: 196   # 14*14

# LLM
llm_name: "Qwen/Qwen2.5-0.5B"
lora_rank: 32
lora_alpha: 64
lora_targets: ["q_proj", "k_proj", "v_proj", "o_proj"]

# Action head (GR00T)
d_a: 1024
n_heads_action: 16
num_layers_action: 16
ffn_ratio_action: 4
flow_steps_inference: 10
beta_alpha: 1.5
beta_beta: 1.0

# Aux head
d_aux: 768
n_heads_aux: 12
num_layers_aux: 12         # <-- 12 层
ffn_ratio_aux: 4
aux_T: 4
aux_H: 14
aux_W: 14

# Loss 调度
lambda_aux_init:  1.0
lambda_aux_final: 0.2
warmup_ratio:     0.1
```

---

## 14. 实现注意事项(给 Claude Code)

### 14.1 V-JEPA Encoder 加载

- 从 HuggingFace 加载:`facebook/vjepa2-vitl-fpc64-256` 或其他权重
- 确认 encoder 输出是 `[B, N_tokens, D]` 格式(不是 `[B, D, T, H, W]`)
- 如果官方接口返回 dict,提取 `last_hidden_state` 或类似字段
- **必须验证**:输入 8 帧 `[B, 8, 3, 224, 224]` 时输出 `[B, 4*14*14=784, 1024]`

### 14.2 LLM LoRA 配置

- 用 PEFT 库的 `LoraConfig`,`target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`
- 确认 Qwen2.5-0.5B 的 attention 模块命名(可能是 `c_attn` 或分开的 q/k/v)
- `task_type` 设为 `CAUSAL_LM` 或不设(纯 feature extractor 模式)

### 14.3 输入 embedding 接口

- 通过 `llm.get_input_embeddings()` 得到 embedding 层
- 或 `llm.model.embed_tokens`(取决于 Qwen 的具体接口)
- 用 `inputs_embeds=` 参数传入拼接好的嵌入,不要走 `input_ids`

### 14.4 Hidden state 提取

- `output_hidden_states=True` 才会返回所有层的 hidden
- `hidden_states[-1]` 是最后一层输出
- 确认 shape 是 `[B, L, D_llm]`

### 14.5 Attention Mask

- 视觉 token 没有 pad,language token 在 batch 内可能需要 pad
- 构造完整的 attention mask `[B, L]`,1 = 有效,0 = padding
- 传给 LLM 时正确设置 `attention_mask` 参数

### 14.6 初始化

- 所有 projection(query_proj, memory_proj, output_proj, action_proj):`nn.init.xavier_uniform_` 或 LLM 默认
- Position embeddings:`torch.randn * 0.02`
- LayerNorm 默认即可
- Action head 的最后 output layer 可以用 zero init(稳定初期训练)

### 14.7 训练顺序建议

按以下顺序调试:
1. **只跑 action head**(λ_aux = 0),验证基本 policy 能训起来
2. **加 aux head**(λ_aux 按调度),观察 aux loss 是否下降、action loss 是否受影响
3. **消融实验**:关掉 aux head 和带 aux head 分别跑,对比 success rate

### 14.8 显存考虑

- 单卡 24GB 大概能跑 batch_size=4~8(取决于 LoRA 和 sequence 长度)
- Aux head 的 cross-attention map `[B, n_heads, N_q, L]` 在 V=3 时很大,考虑用 flash attention
- 梯度累积可以用来模拟大 batch

### 14.9 关键单元测试

先写这几个测试再跑完整训练:

1. `test_vjepa_shape`:确认 V-JEPA 输入 2 帧和 8 帧分别得到 1×14×14 和 4×14×14 tokens
2. `test_vision_projector`:shape 对齐
3. `test_llm_forward`:LLM hidden state shape 正确
4. `test_action_head_forward`:训练模式输出 loss 和 velocity,推理模式输出 action
5. `test_aux_head_forward`:输出 shape 是 `[B, V, 4, 14, 14, D_jepa]`
6. `test_end_to_end_loss`:整个模型 backward 不报错

---

## 15. 变更清单(相对 VLA-Adapter)

实现时需要**替换/新增**的部分:

| 变更 | 说明 |
|---|---|
| 视觉 backbone | DINOv2+SigLIP → V-JEPA 2 ViT-L(替换) |
| Vision projector 输入维度 | 2048 → 1024(单个 encoder) |
| ActionQuery + Bridge Attention | **移除**(不使用 VLA-Adapter 的 per-layer 读取机制) |
| Action head | OFT 风格 MLP → **GR00T 16 层交替 self/cross + flow matching** |
| Aux head | **新增**(12 层 cross-attention decoder) |
| Aux target 生成 | **新增**(用同一个 V-JEPA encoder 编码未来帧) |
| Loss | 单一 action loss → action loss + λ × aux loss |
| Batch 数据 | 原 batch → 额外加 `future_frames` 字段 |

保留不变的部分:
- 整体 Prismatic-VLM 骨架(视觉 token + 语言 token 拼接 → LLM)
- 视觉在前、语言在后的顺序
- LLM 冻结 + LoRA 的训练策略
- Causal attention(无需特殊 mask)
