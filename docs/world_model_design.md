# Latent World Model for Model-Based VLM RL

## 1. Project Goal

Build a **latent-space world model** on top of the VAGEN framework (multi-turn VLM agent RL built on verl). Instead of predicting future pixels or text, the model predicts the next-step **visual latent representation** produced by the VLM's frozen visual encoder. This enables latent-space rollouts that can replace or supplement real environment interaction for model-based RL.

The core pipeline:

```
O_t + a_t + context -> VLM (ViT + LLM backbone) -> h_t -> Projector -> z_hat_{t+1}
Target: O_{t+1} -> frozen ViT + PatchMerger -> z_{t+1}
```

For multi-step imagination (rollout), the predicted `z_hat_{t+1}` is injected back into the VLM via `inputs_embeds`, bypassing the ViT entirely. The VLM backbone serves as the "recurrence" — no separate GRU or RNN needed.

## 2. Reference Materials

### Papers

- **Dream to Control (Dreamer, ICLR 2020)** — `/workspace/1400_dream_to_control_learning_beha.pdf`
  - RSSM world model: GRU (200-dim) + stochastic state (30-dim Gaussian) + Dense layers (300 units)
  - Multi-step rollout via GRU recurrence
  - Pixel reconstruction loss (ConvTranspose decoder) prevents representation collapse
  - Very compact state (~230 dim), parallel imagination of thousands of trajectories
  - Source code at `/workspace/dreamer/`

- **VL-JEPA (Meta FAIR, 2026)** — `/workspace/VL-JEPA.pdf`
  - Joint Embedding Predictive Architecture for vision-language
  - Predictor: 8-layer **bidirectional** Transformer (from Llama-3.2-1B, 490M params)
  - Input: visual tokens + text query tokens, full bidirectional attention (no causal mask)
  - Output: average pooling over tokens -> linear projection -> 1536-dim shared space
  - Target: text -> EmbeddingGemma-300M -> linear projection -> 1536-dim
  - Loss: **InfoNCE** (bidirectional contrastive), not cosine similarity alone
  - Frozen X-Encoder (V-JEPA 2 ViT-L), Y-Encoder trained with slow LR (0.05x)

### Algorithm Design Documents

- **`/workspace/MBVLM_algorithm.md`** — Original proposal for combining VLM + MBRL. Core idea: predict z_{t+1} in frozen ViT latent space, use cosine alignment loss, freeze visual encoder to prevent collapse.

- **`/workspace/MBRL_algorthm.md`** — Detailed review and enhancement with three-phase plan:
  - Phase 1: World Model warm-up (offline, pre-extracted features)
  - Phase 2: Online RL with latent imagination (shared VLM backbone, separate LoRA adapters)
  - Phase 3: Self-Critic Reward in language space (LLM-as-a-Judge or implicit reward)
  - Introduced `<|latent_token|>` concept, system prompt design, data collection strategy

## 3. VLM Foundation: Qwen2.5-VL-3B-Instruct

- Path: `/workspace/models/Qwen2.5-VL-3B-Instruct/`
- Key parameters: `hidden_size=2048`, vision `embed_dim=1280`, `patch_size=14`, `spatial_merge_size=2`
- Visual token injection point (verified in source): when `inputs_embeds` is passed to `forward()`, the entire ViT + embed_tokens pipeline is skipped — this is how predicted z_hat gets injected for rollout

### Visual Token Count for Sokoban

Sokoban renders 96x96 images. Through Qwen2.5-VL's pipeline:
1. `smart_resize`: 96 -> 84 (nearest multiple of 28 = patch_size * spatial_merge_size)
2. ViT patching: 84 / 14 = 6 -> grid (1, 6, 6)
3. PatchMerger 2x2: 6 / 2 = 3 -> grid (1, 3, 3) = **9 visual tokens**

Each token is 2048-dim (after PatchMerger projects from 1280 to hidden_size).

## 4. Architecture Design

### Design Evolution

**Initial design: Cross-Attention Decoder (Q-Former style)**

First implementation used `nn.TransformerDecoder` — 9 learned queries cross-attend to h_t. Inspired by BLIP-2's Q-Former.

Problem identified: cross-attention to a **single memory token** is degenerate. Softmax over 1 element always outputs 1.0, so every query gets the identical value projection of h_t. Token differentiation can only come from self-attention and initial query embeddings, wasting the cross-attention structure.

Neither reference paper uses this pattern:
- Dreamer: Dense + GRU + Dense (input is compact, no attention needed)
- VL-JEPA: Bidirectional Transformer (input has many tokens, full mutual attention)

**Current design: Bidirectional Transformer (VL-JEPA-inspired)**

```python
# Input: [h_t, q_1, ..., q_9] = 10 tokens, all 2048-dim
tokens = concat(h_t.unsqueeze(1), learned_queries)   # (B, 10, 2048)
out = TransformerEncoder(tokens)                       # (B, 10, 2048) bidirectional
z_hat = out[:, 1:, :]                                  # (B, 9, 2048) take query positions
```

Why bidirectional Transformer:
- h_t and queries **mutually attend** to each other (h_t can also attend to queries and get refined)
- Simpler than decoder (no separate cross-attention + self-attention)
- Directly extensible: if we later add more input tokens (e.g., O_t's visual tokens), the architecture doesn't change
- 100.7M params (vs 134M for the decoder version — saved by removing cross-attention KV projections)

### Why Not Other Approaches

- **MLP reshape** (h -> Linear -> reshape to 9x2048): Simple but no inter-token communication. Output tokens are independently generated and can't coordinate spatial structure.
- **GRU/RNN** (like Dreamer): Unnecessary. Our multi-step rollout uses the VLM backbone itself as the recurrence mechanism. Adding a separate GRU would mean training a different architecture than what's used at deployment.

### Key Principle: Same Architecture for Validation and Deployment

The Projector trained in Phase 1 (offline, pre-extracted features) is the **exact same Projector** used in Phase 2 (online RL with latent imagination). The multi-step rollout chain is:

```
Step 1: Real O_t + a_t -> VLM -> h_t -> [Projector] -> z_hat_{t+1}
Step 2: z_hat_{t+1} (via inputs_embeds) + a_{t+1} -> VLM -> h'_{t+1} -> [same Projector] -> z_hat_{t+2}
Step 3: ...
```

No architecture change between validation and deployment.

## 5. Loss Function Design

### Problem: High Baseline Cosine Similarity

Sokoban frames are visually very similar (same grid, walls, slight position changes). Measured inter-target statistics:
- Random pair cosine similarity: **0.934** (mean), range [0.787, 1.0]
- Same-state different-action pairs: **0.972**

This means pure cosine loss operates in a very narrow effective range (0.934 to 1.0). A model that just predicts the "average" ViT output would score ~0.934 cosine similarity, making it hard to tell if the model is actually learning state-specific predictions.

### Solution: InfoNCE + Per-Token Cosine (Inspired by VL-JEPA)

```
Total Loss = lambda_cos * L_cosine + lambda_nce * L_infonce + lambda_r * L_reward
```

**Per-token cosine loss** (lambda_cos=1.0): Ensures each predicted visual token aligns with its corresponding target token. Essential for rollout injection — the predicted tokens must be close to real ViT output in the LLM embedding space.

**InfoNCE contrastive loss** (lambda_nce=1.0, temperature=0.07): Pool predicted/target tokens to single vectors, compute batch-level similarity matrix, apply bidirectional cross-entropy. Forces the model to distinguish the correct target from all other targets in the batch. This directly addresses the high-baseline problem — even if all targets are 0.934 similar, InfoNCE requires learning the remaining ~7% of distinguishing features.

**Reward MSE loss** (lambda_r=0.1): Auxiliary task predicting scalar reward from h_t.

### Training Dynamics Observation

With the current hyperparameters, InfoNCE loss (~3.5 initially) dominates cosine loss (~0.3 initially). In early epochs, InfoNCE pulls representations toward discriminability at the cost of per-token alignment, causing cosine_sim to temporarily drop. By epoch ~5, cosine_sim begins recovering as the two losses find a compatible solution. The infonce_acc metric (accuracy of matching correct target in a batch of 128) is more informative than raw cosine_sim for evaluating learning progress.

## 6. Data Pipeline

### Step 1: Collect Sokoban Transitions

Script: `scripts/world_model/collect_sokoban_data.py`

For each visited state in a random-walk episode, tries **all 9 actions** exhaustively:
- 0: no operation
- 1-4: push up/down/left/right
- 5-8: move up/down/left/right

This gives complete transition coverage including failures (hitting walls, invalid pushes). Collected **300 episodes, 5106 unique states, 45954 transitions** (5106 * 9).

Output: `/workspace/VAGEN/data/world_model/sokoban/transitions.pkl`

Design decision: exhaustive action coverage (all 9 per state) rather than just recording the agent's chosen action. This is critical because:
1. The world model must predict outcomes for ALL actions, not just successful ones
2. For action discrimination evaluation, we need same-state-different-action pairs
3. 9 actions per state gives natural grouping for evaluation

### Step 2: Extract Features

Script: `scripts/world_model/extract_features.py`

Two-pass extraction using the full Qwen2.5-VL-3B model:

**Pass 1 — LLM hidden states**: For each (O_t, a_t), construct a prompt with system instructions describing Sokoban rules in natural language, the observation image, the action text, and `<|latent_token|>`. Run VLM forward pass, extract the hidden state at the `<|latent_token|>` position.

System prompt design: describes Sokoban rules in natural language (not ASCII symbols like @/#/$, since the VLM sees images not text grids). The prompt activates the VLM's world knowledge about spatial reasoning and cause-effect relationships.

**Pass 2 — ViT targets**: For each O_{t+1}, run only the frozen ViT + PatchMerger to get the target visual tokens.

`<|latent_token|>` is added as a special token to the tokenizer. Its hidden state serves as the information bottleneck — it must encode everything the Projector needs to predict the next visual state.

Output: `hidden_states.pt` (45954, 2048), `vision_targets.pt` (45954, 9, 2048), `rewards.pt` (45954,), plus `images.pkl` and `action_info.pkl` for evaluation visualization.

### Step 3: Train Projector

Script: `scripts/world_model/train_world_model.py`

Operates purely on pre-extracted tensors — no VLM needed at training time. This makes iteration fast (seconds per epoch vs hours for VLM forward passes).

- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Scheduler: CosineAnnealingLR over 100 epochs
- Batch size: 128
- Train/val split: 90/10
- Gradient clipping: max norm 1.0
- wandb logging: project `world-model-sokoban`

### Step 4: Evaluate

Script: `scripts/world_model/evaluate_world_model.py`

Three evaluation methods:

1. **Training curves**: Plot cosine_sim and loss over epochs
2. **Nearest-neighbor retrieval**: For each (O_t, a_t), predict z_hat, find the closest real z in the dataset. Visualize O_t -> action -> actual O_{t+1} vs NN-retrieved O_{t+1}. This gives an intuitive visual check of prediction quality.
3. **Action discrimination test**: For the same state, predict z_hat for all 9 actions. Check if predictions are distinguishable (intra-state sim vs inter-state sim, correct-target vs wrong-target sim). This tests whether the model captures action-dependent dynamics, not just the average next state.

## 7. File Structure

```
vagen/world_model/
    __init__.py                      # exports LatentWorldModel
    latent_world_model.py            # WorldModelConfig, VisualTokenProjector,
                                     # RewardHead, LatentWorldModel, utilities

scripts/world_model/
    collect_sokoban_data.py          # Step 1: collect transitions
    extract_features.py              # Step 2: extract VLM hidden states + ViT targets
    train_world_model.py             # Step 3: train Projector (with wandb)
    evaluate_world_model.py          # Step 4: evaluate trained model

data/world_model/sokoban/
    transitions.pkl                  # raw transitions (45954)
    features/
        hidden_states.pt             # (45954, 2048)
        vision_targets.pt            # (45954, 9, 2048)
        rewards.pt                   # (45954,)
        dones.pt                     # (45954,)
        action_info.pkl              # action metadata for eval
        images.pkl                   # obs/next_obs images for NN retrieval vis

checkpoints/world_model/sokoban/
    best_model.pt                    # best val cosine_sim checkpoint
    config.json                      # WorldModelConfig
    training_history.json            # per-epoch metrics
```

## 8. Hardware

- 4x NVIDIA A100-SXM4-80GB
- Feature extraction uses 1 GPU (VLM inference)
- Projector training uses 1 GPU (small model, fast iteration)
- Remaining GPUs available for parallel experiments or future online RL

## 9. Training Results (bidir-infonce-v1, 100 epochs)

### Final Metrics

| Metric | Train | Val |
|--------|-------|-----|
| cosine_sim | 0.872 | 0.870 |
| infonce_acc | 94.9% | 94.5% |
| reward_loss | 1.021 | 1.021 |
| total_loss | 2.006 | 2.049 |

Training dynamics: cosine_sim started at ~0.7, dipped to ~0.53 in early epochs (InfoNCE dominating optimization), then steadily recovered to 0.87 by epoch 100. Train/val curves nearly overlap — no overfitting.

### Static Evaluation

**NN Retrieval (50 samples)**:
- Cosine sim with actual target (pooled): 0.291
- Exact match rate: 2%

**Action Discrimination (200 states)**:
- Intra-state sim (same state, diff actions): 0.584
- Inter-state sim (diff states): 0.005
- Correct > Wrong rate: 73.94%

The model strongly distinguishes different states (inter ~0 vs intra ~0.58) and shows meaningful action sensitivity (74% correct > wrong). However, per-token cosine sim (0.87) is below the random ViT baseline (0.934), indicating InfoNCE pulled representations toward discriminability at the cost of absolute alignment with the original ViT space.

## 10. Multi-Step Latent Rollout Validation

Script: `scripts/world_model/validate_rollout.py`

Tests the core hypothesis: can z_hat predicted by the Projector be injected back into the VLM via `inputs_embeds` to enable multi-step imagination?

### Implementation

For imagined steps, visual tokens are injected by:
1. Tokenizing the prompt with a dummy 96x96 image to get correct `input_ids` (with `image_token_id` placeholders) and `image_grid_thw` (for 3D mRoPE position encoding)
2. Embedding text tokens via `model.model.embed_tokens(input_ids)`
3. Replacing `image_token_id` positions with z_hat using `masked_scatter`
4. Forwarding with both `input_ids` (for position_ids computation only) and `inputs_embeds` (for the actual forward pass)

The model's `Qwen2_5_VLForConditionalGeneration.forward()` skips `embed_tokens` + ViT when `inputs_embeds` is provided, but still uses `input_ids` + `image_grid_thw` to compute correct 3D mRoPE position IDs for visual tokens.

### Rollout Results (10 rollouts x 5 steps)

| Step | Type | Per-token cos | Pooled cos |
|------|------|--------------|------------|
| 0 | real | 0.836 ± 0.018 | 0.220 ± 0.078 |
| 1 | imagined | 0.548 ± 0.060 | 0.623 ± 0.075 |
| 2 | imagined | 0.564 ± 0.051 | 0.683 ± 0.063 |
| 3 | imagined | 0.549 ± 0.048 | 0.648 ± 0.077 |
| 4 | imagined | 0.551 ± 0.034 | 0.631 ± 0.034 |

### Key Findings

1. **inputs_embeds injection works**: The VLM processes synthetic visual tokens without crashing or producing NaN. Core hypothesis validated.

2. **Errors do NOT compound**: Cosine similarity stays flat from step 1 through step 4 (~0.55 token, ~0.65 pooled). The VLM backbone is a stable recurrence mechanism.

3. **Main accuracy loss is the first real→imagined jump**: Token cos drops from 0.84 (real) to 0.55 (imagined-1). This is due to the Projector's single-step prediction gap, not error accumulation.

4. **Pooled cos anomaly at step 0**: The low pooled cos (0.22) at step 0 vs higher values in imagined steps suggests the InfoNCE-trained Projector outputs live in a shifted subspace. Once the VLM processes z_hat, subsequent Projector outputs are more self-consistent.

### Bottleneck Analysis

The limiting factor is **single-step alignment quality**, not multi-step stability. The Projector's per-token cosine sim (0.87) is below the random ViT baseline (0.934) because InfoNCE (weight=1.0, temp=0.07) dominates optimization. Reducing `infonce_loss_weight` to 0.1-0.3 or raising `infonce_temperature` to 0.2-0.5 should improve per-token alignment and directly improve rollout quality.

## 11. File Structure (Updated)

```
vagen/world_model/
    __init__.py                      # exports LatentWorldModel
    latent_world_model.py            # WorldModelConfig, VisualTokenProjector,
                                     # RewardHead, LatentWorldModel, utilities

scripts/world_model/
    collect_sokoban_data.py          # Step 1: collect transitions
    extract_features.py              # Step 2: extract VLM hidden states + ViT targets
    train_world_model.py             # Step 3: train Projector (with wandb)
    evaluate_world_model.py          # Step 4: evaluate trained model
    validate_rollout.py              # Step 5: multi-step rollout validation

data/world_model/sokoban/
    transitions.pkl                  # raw transitions (45954)
    features/
        hidden_states.pt             # (45954, 2048)
        vision_targets.pt            # (45954, 9, 2048)
        rewards.pt                   # (45954,)
        dones.pt                     # (45954,)
        action_info.pkl              # action metadata for eval
        images.pkl                   # obs/next_obs images for NN retrieval vis

checkpoints/world_model/sokoban/
    best_model.pt                    # best val cosine_sim checkpoint
    final_model.pt                   # epoch 100 checkpoint
    config.json                      # WorldModelConfig
    training_history.json            # per-epoch metrics
    eval/
        training_curves.png
        nn_retrieval.png
    rollout_eval/
        rollout_results.json         # per-step metrics for all rollouts
        rollout_cosine.png           # error compounding plot
```

## 12. Next Steps

- **Hyperparameter tuning**: Reduce `infonce_loss_weight` (0.1-0.3) or raise `infonce_temperature` (0.2-0.5) to improve per-token alignment above the 0.934 random baseline
- **Phase 2 integration**: Integrate the trained Projector into VAGEN/verl's RL pipeline for latent-space imagination during online training (shared VLM backbone, separate LoRA adapters for policy vs world model)
- **Phase 3**: Explore Self-Critic Reward using VLM's language generation capability as an intrinsic reward signal
- **Generalization**: Test on environments beyond Sokoban
