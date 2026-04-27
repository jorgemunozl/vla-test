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