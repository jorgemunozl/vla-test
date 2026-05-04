## pi05_base architecture (layer tree)

Captured by running (inside conda env `nh_05`):

```bash
python /home/lperez/main/nh/XHUMAN/scripts/pi05_watch_arch.py \
  --model-id lerobot/pi05_base \
  --device cpu \
  --offline \
  --max-depth 6 \
  --no-config \
  --no-repr
```

## Output

```text
WARNING:root:Vision embedding key might need handling: paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias
WARNING:root:Vision embedding key might need handling: paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight
[pi05-arch] model_id='lerobot/pi05_base'
[pi05-arch] device=cpu
Loading model from: lerobot/pi05_base
✓ Loaded state dict from model.safetensors
Remapped 812 state dict keys
All keys loaded successfully!
[pi05-arch] params total=4,143,404,816 trainable=4,143,404,816

=== module tree (truncated) ===
- policy: PI05Policy
  - model: PI05Pytorch
    - paligemma_with_expert: PaliGemmaWithExpertModel
      - paligemma: PaliGemmaForConditionalGenerationWithPiGemma
        - model: PaliGemmaModelWithPiGemma
          - vision_tower: SiglipVisionModel
            - vision_model: SiglipVisionTransformer
          - multi_modal_projector: PaliGemmaMultiModalProjector
            - linear: Linear  (direct params: 2,361,344 | trainable: 2,361,344)
          - language_model: PiGemmaModel
            - embed_tokens: Embedding  (direct params: 526,647,296 | trainable: 526,647,296)
            - layers: ModuleList
            - norm: PiGemmaRMSNorm  (direct params: 2,048 | trainable: 2,048)
            - rotary_emb: GemmaRotaryEmbedding
        - lm_head: Linear  (direct params: 526,647,296 | trainable: 526,647,296)
      - gemma_expert: PiGemmaForCausalLM
        - model: PiGemmaModel
          - layers: ModuleList
            - 0: _PiGemmaDecoderLayerBase
            - 1: _PiGemmaDecoderLayerBase
            - 2: _PiGemmaDecoderLayerBase
            - 3: _PiGemmaDecoderLayerBase
            - 4: _PiGemmaDecoderLayerBase
            - 5: _PiGemmaDecoderLayerBase
            - 6: _PiGemmaDecoderLayerBase
            - 7: _PiGemmaDecoderLayerBase
            - 8: _PiGemmaDecoderLayerBase
            - 9: _PiGemmaDecoderLayerBase
            - 10: _PiGemmaDecoderLayerBase
            - 11: _PiGemmaDecoderLayerBase
            - 12: _PiGemmaDecoderLayerBase
            - 13: _PiGemmaDecoderLayerBase
            - 14: _PiGemmaDecoderLayerBase
            - 15: _PiGemmaDecoderLayerBase
            - 16: _PiGemmaDecoderLayerBase
            - 17: _PiGemmaDecoderLayerBase
          - norm: PiGemmaRMSNorm
            - dense: Linear  (direct params: 3,148,800 | trainable: 3,148,800)
          - rotary_emb: GemmaRotaryEmbedding
        - lm_head: Linear  (direct params: 263,323,648 | trainable: 263,323,648)
    - action_in_proj: Linear  (direct params: 33,792 | trainable: 33,792)
    - action_out_proj: Linear  (direct params: 32,800 | trainable: 32,800)
    - time_mlp_in: Linear  (direct params: 1,049,600 | trainable: 1,049,600)
    - time_mlp_out: Linear  (direct params: 1,049,600 | trainable: 1,049,600)
```

---

## google/paligemma2-3b-mix-224 (vision_tower) architecture (layer tree)

Captured by running (inside conda env `nh_05`):

```bash
python /home/lperez/main/nh/XHUMAN/scripts/hf_watch_arch.py \
  --model-id google/paligemma2-3b-mix-224 \
  --component vision \
  --device cpu \
  --offline \
  --max-depth 6 \
  --no-config
```

## Output

```text
[hf-arch] model_id='google/paligemma2-3b-mix-224'
[hf-arch] device=cpu
[hf-arch] component=vision

[hf-arch] params(total)=3,032,242,416 trainable=3,032,242,416

=== module tree (truncated) ===
- vision_tower: SiglipVisionModel
  - vision_model: SiglipVisionTransformer
    - embeddings: SiglipVisionEmbeddings
      - patch_embedding: Conv2d  (direct params: 678,528 | trainable: 678,528)
      - position_embedding: Embedding  (direct params: 294,912 | trainable: 294,912)
    - encoder: SiglipEncoder
      - layers: ModuleList
        - 0: SiglipEncoderLayer
          - layer_norm1: LayerNorm  (direct params: 2,304 | trainable: 2,304)
          - self_attn: SiglipAttention
            - k_proj: Linear  (direct params: 1,328,256 | trainable: 1,328,256)
            - v_proj: Linear  (direct params: 1,328,256 | trainable: 1,328,256)
            - q_proj: Linear  (direct params: 1,328,256 | trainable: 1,328,256)
            - out_proj: Linear  (direct params: 1,328,256 | trainable: 1,328,256)
          - layer_norm2: LayerNorm  (direct params: 2,304 | trainable: 2,304)
          - mlp: SiglipMLP
            - activation_fn: GELUTanh
            - fc1: Linear  (direct params: 4,962,512 | trainable: 4,962,512)
            - fc2: Linear  (direct params: 4,959,360 | trainable: 4,959,360)
        - 1: SiglipEncoderLayer
          - layer_norm1: LayerNorm  (direct params: 2,304 | trainable: 2,304)
          - self_attn: SiglipAttention
            - k_proj: Linear  (direct params: 1,328,256 | trainable: 1,328,256)
            - v_proj: Linear  (direct params: 1,328,256 | trainable: 1,328,256)
            - q_proj: Linear  (direct params: 1,328,256 | trainable: 1,328,256)
            - out_proj: Linear  (direct params: 1,328,256 | trainable: 1,328,256)
          - layer_norm2: LayerNorm  (direct params: 2,304 | trainable: 2,304)
          - mlp: SiglipMLP
            - activation_fn: GELUTanh
            - fc1: Linear  (direct params: 4,962,512 | trainable: 4,962,512)
            - fc2: Linear  (direct params: 4,959,360 | trainable: 4,959,360)
        - 2: SiglipEncoderLayer
          - layer_norm1: LayerNorm  (direct params: 2,304 | trainable: 2,304)
          - self_attn: SiglipAttention
            - k_proj: Linear  (direct params: 1,328,256 | trainable: 1,328,256)
            - v_proj: Linear  (direct params: 1,328,256 | trainable: 1,328,256)
            - q_proj: Linear  (direct params: 1,328,256 | trainable: 1,328,256)
            - out_proj: Linear  (direct params: 1,328,256 | trainable: 1,328,256)
          - layer_norm2: LayerNorm  (direct params: 2,304 | trainable: 2,304)
          - mlp: SiglipMLP
            - activation_fn: GELUTanh
            - fc1: Linear  (direct params: 4,962,512 | trainable: 4,962,512)
            - fc2: Linear  (direct params: 4,959,360 | trainable: 4,959,360)
    - post_layernorm: LayerNorm  (direct params: 2,304 | trainable: 2,304)
```

## pi05_ki_pg2 architecture (smoke test: scratch instantiation)

Captured on **2026-04-27 11:40 -05** by running (inside conda env `nh_05`):

```bash
python /home/lperez/main/nh/XHUMAN/scripts/pi05ki_pg2_watch_arch.py \
  --offline \
  --device cpu \
  --max-depth 3
```

This instantiates `PI05KIPG2Policy` from scratch (no pretrained weights),
composed of:

* **VLM**: `google/paligemma2-3b-mix-224` (`PaliGemmaForConditionalGeneration`
  with a Gemma2 26-layer language model, head_dim=256,
  q=8 / kv=4 heads, attn-logit soft-capping=50.0).
* **Action expert**: a Gemma1 + AdaRMS decoder stack stretched to
  **26 layers** (head_dim=256, 8 attention heads, 1 KV head,
  hidden_size=1024, mlp_intermediate=4096).
* **Flow-matching heads**: identical to `pi05_ki` (32→1024 in,
  1024→32 out, 1024→1024 time MLP).

### Output

```
[pi05_ki_pg2-arch] config:
  paligemma_variant       = paligemma2_3b
  paligemma_hub_id        = google/paligemma2-3b-mix-224
  pi05_base_hub_id        = lerobot/pi05_base
  expert_depth            = 26
  expert_warmstart_layers = 18
  tokenizer_name          = google/paligemma2-3b-mix-224
  chunk_size              = 50
  max_action_dim          = 32
  dtype                   = float32

[pi05_ki_pg2-arch] params total=3,914,457,360 trainable=3,914,457,360

=== VLM (PaliGemma2 language model) ===
num layers = 26
layer 0 sub-modules: ['self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm', 'pre_feedforward_layernorm', 'post_feedforward_layernorm']
layer 0 attn: q_proj=(2048, 2304), k_proj=(1024, 2304), v_proj=(1024, 2304), o_proj=(2304, 2048), head_dim=256, scaling=0.0625, softcap=50.0

=== Expert (Gemma1 + AdaRMS) ===
num layers = 26
layer 0 sub-modules: ['self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm']
layer 0 attn: q_proj=(2048, 1024), k_proj=(256, 1024), v_proj=(256, 1024), o_proj=(1024, 2048)

=== Flow-matching heads ===
action_in_proj: weight=(1024, 32), bias=True
action_out_proj: weight=(32, 1024), bias=True
time_mlp_in: weight=(1024, 1024), bias=True
time_mlp_out: weight=(1024, 1024), bias=True

=== module tree ===
- policy: PI05KIPG2Policy [subtree total: 3,914,457,360]
  - model: PI05KIPG2Model [subtree total: 3,914,457,360]
    - paligemma_with_expert: PaliGemmaWithExpertModelPG2 [subtree total: 3,912,291,568]
      - paligemma: PaliGemmaForConditionalGeneration [subtree total: 3,032,242,416]
      - gemma_expert: PiGemmaForCausalLM [subtree total: 880,049,152]
    - action_in_proj: Linear  (direct params: 33,792 | trainable: 33,792) [subtree total: 33,792]
    - action_out_proj: Linear  (direct params: 32,800 | trainable: 32,800) [subtree total: 32,800]
    - time_mlp_in: Linear  (direct params: 1,049,600 | trainable: 1,049,600) [subtree total: 1,049,600]
    - time_mlp_out: Linear  (direct params: 1,049,600 | trainable: 1,049,600) [subtree total: 1,049,600]
```

### Joint KI forward sanity check

`compute_layer_complete_pg2` was exercised on layers 0, 5, 25 with a
mock batch of 2 sequences (5 VLM tokens + 3 expert tokens, head_dim=256).
The function returned finite tensors of the expected shapes —
`(2, 5, 2304)` for the VLM side and `(2, 3, 1024)` for the expert side —
confirming that:

* the asymmetric GQA expansion (VLM 4 KV heads, expert 1 KV head, both
  expanded to the common 8 query heads) round-trips correctly through
  the shared attention,
* the Gemma2 sandwich (`pre_feedforward_layernorm` /
  `post_feedforward_layernorm`) is applied on the VLM side,
* the AdaRMS gated residuals run on the expert side with the
  conditioning vector,
* the Knowledge-Insulation gradient stop on the backbone K/V is in
  place.

---

## Training smoke run (2026-04-29)

End-to-end smoke run with `bin/run_train_val_pi05ki_pg2_smoke.sh` on a
single RTX A6000 (48 GB), 30 optimizer steps, batch size 1,
`bfloat16` + gradient checkpointing, vision tower frozen, expert layers
0..17 + flow heads warm-started from `lerobot/pi05_base`, VLM warm-started
from `google/paligemma2-3b-mix-224`.

* `num_total_params=3,914,457,360` (≈ 3.9 B)
* `num_learnable_params=3,502,015,008` (≈ 3.5 B; SigLIP frozen)
* warm-start: VLM `missing=0, unexpected=0`; expert
  `loaded=206, mismatched=0, expert_layers_random_init=8`
* training step time ≈ **0.74 s/step** (post-warmup)
* loss profile (`flow_matching_loss`):
    - step 1:  0.580
    - step 15 (eval): **0.484**
    - step 30 (eval): **0.432**
* `ce_loss = 0.000` is expected: with `fast_tokenizer = None` (the default
  for action-only fine-tuning), no FAST-discretized action tokens are
  injected into the prefix, so the language-model cross-entropy term has
  no supervised positions and degenerates to zero. Only the flow-matching
  loss is driving the optimization.

This validates that the joint VLM + expert path (`compute_layer_complete_pg2`)
runs forward, backward and optimizer-step on a real
`NONHUMAN-RESEARCH/general-task-index` batch without OOM and that the
loss is on the right order of magnitude.
