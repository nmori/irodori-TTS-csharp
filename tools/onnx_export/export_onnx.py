"""
Export an Irodori-TTS v4 checkpoint + DACVAE codec to ONNX for IrodoriTts.Onnx (WebGPU backend).

Usage:
    python export_onnx.py --irodori-repo <path/to/Irodori-TTS> \
        --checkpoint <model.safetensors> --codec <weights.pth> --out <output_dir>

Output directory layout (what IrodoriTts.Onnx expects):
    irodori_onnx.json            model / codec / inference configuration
    tokenizer/                   copied from the checkpoint folder
    text_embedding.safetensors   ModernBERT token embedding table (gathered on the CPU)
    text_encoder.onnx            ModernBERT + text (+ caption) projector + norm      (fp32)
    speaker_encoder.onnx         patched reference latent -> speaker state            (fp32)
    duration.onnx                duration predictor                                   (fp32)
    dit_context.onnx             per-layer conditioning K/V of the DiT, once per request (same precision as dit_step_kv)
    dit_step_kv.onnx             one velocity evaluation over those K/V  (fp32 for MeanFlow, fp16 for RF by default)
    dit_step.onnx                legacy single-graph step (only with --legacy-dit-step)
    dacvae_encoder.onnx          waveform -> posterior-mean latent                    (fp32)
    dacvae_decoder.onnx          latent -> waveform                                   (fp32)
    dacvae_decoder_fp16.onnx     the same in fp16 (float32 in / out), used on WebGPU  (unless --no-decoder-fp16)

Only pretrained-text-encoder checkpoints (v4 family) are supported.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch


BOOL_UNSAFE_OPS = {
    "Slice", "ScatterElements", "ScatterND", "Gather", "GatherElements", "GatherND", "Pad", "Tile",
    "Unsqueeze", "Squeeze", "Reshape", "Expand", "Transpose", "Flatten", "Concat", "Split", "Identity",
}
# Notes on WebGPU EP robustness (ONNX Runtime 1.30 + EP.WebGpu 0.4.0), all observed as hard aborts:
#  - data-movement ops on bool tensors, and bool graph inputs shared by CPU and GPU nodes
#  - gathers from tensors larger than the adapter's storage-buffer binding limit


def patch_bool_ops(path: Path) -> int:
    """
    The WebGPU EP aborts the process (0xC0000409) on data-movement ops applied to bool tensors
    (observed for Slice and Unsqueeze). Rewrite them as Cast(bool->int32) -> op -> Cast(->bool).
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(path), load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(model)
    types = {}
    for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        types[v.name] = v.type.tensor_type.elem_type
    for init in model.graph.initializer:
        types[init.name] = init.data_type

    def data_inputs(node):
        if node.op_type == "Concat":
            return list(range(len(node.input)))
        if node.op_type.startswith("Scatter"):
            return [0, 2]
        return [0]

    used = {n for node in model.graph.node for n in list(node.input) + list(node.output) + [node.name]}
    used |= {i.name for i in model.graph.initializer} | {i.name for i in model.graph.input}

    def fresh(base):
        k = 0
        while f"{base}__i32_{k}" in used:
            k += 1
        name = f"{base}__i32_{k}"
        used.add(name)
        return name

    new_nodes, patched = [], 0
    for node in model.graph.node:
        if node.op_type not in BOOL_UNSAFE_OPS:
            new_nodes.append(node)
            continue
        idx = [i for i in data_inputs(node) if i < len(node.input) and node.input[i] and types.get(node.input[i]) == TensorProto.BOOL]
        if not idx:
            new_nodes.append(node)
            continue
        for i in idx:
            cast_name = fresh(node.input[i])
            new_nodes.append(helper.make_node("Cast", [node.input[i]], [cast_name], to=TensorProto.INT32,
                                              name=fresh(f"{node.name}_in{i}")))
            node.input[i] = cast_name
        casts_out = []
        for j, out in enumerate(node.output):
            if not out:
                continue
            tmp = fresh(out)
            node.output[j] = tmp
            casts_out.append(helper.make_node("Cast", [tmp], [out], to=TensorProto.BOOL, name=fresh(f"{node.name}_out{j}")))
        new_nodes.append(node)
        new_nodes.extend(casts_out)
        patched += 1
    # Bool graph inputs/outputs crossing the CPU <-> GPU boundary also abort the WebGPU EP:
    # expose them as int32 (0/1) and convert inside the graph.
    head, tail = [], []
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type != TensorProto.BOOL:
            continue
        inner = fresh(inp.name)
        for node in new_nodes:
            for i, name in enumerate(node.input):
                if name == inp.name:
                    node.input[i] = inner
        head.append(helper.make_node("Cast", [inp.name], [inner], to=TensorProto.BOOL, name=fresh(inp.name + "_to_bool")))
        inp.type.tensor_type.elem_type = TensorProto.INT32
        patched += 1
    for out in model.graph.output:
        if out.type.tensor_type.elem_type != TensorProto.BOOL:
            continue
        inner = fresh(out.name)
        for node in new_nodes:
            for j, name in enumerate(node.output):
                if name == out.name:
                    node.output[j] = inner
            for i, name in enumerate(node.input):
                if name == out.name:
                    node.input[i] = inner
        tail.append(helper.make_node("Cast", [inner], [out.name], to=TensorProto.INT32, name=fresh(out.name + "_to_i32")))
        out.type.tensor_type.elem_type = TensorProto.INT32
        patched += 1
    if patched:
        del model.graph.node[:]
        model.graph.node.extend(head + new_nodes + tail)
        onnx.save(model, str(path))
    return patched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--irodori-repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--codec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dit-precision", choices=["auto", "fp16", "fp32"], default="auto",
                    help="auto: fp32 for MeanFlow checkpoints (4 steps, exact), fp16 for RF checkpoints (40 steps, faster)")
    ap.add_argument("--legacy-dit-step", action="store_true",
                    help="Also export dit_step.onnx (projects the context K/V in every step; for older runtimes)")
    ap.add_argument("--no-decoder-fp16", action="store_true",
                    help="Do not export dacvae_decoder_fp16.onnx (the fp16 decoder the runtime prefers on WebGPU)")
    ap.add_argument("--opset", type=int, default=20)
    ap.add_argument("--only", nargs="*", default=None,
                    help="Export only these graphs (text_encoder speaker_encoder duration dit_context dit_step_kv dit_step dacvae_decoder dacvae_decoder_fp16 dacvae_encoder)")
    args = ap.parse_args()
    sys.path.insert(0, args.irodori_repo)

    import irodori_tts.model as M
    from irodori_tts.config import ModelConfig, merge_dataclass_overrides
    from irodori_tts.inference_runtime import _load_checkpoint_for_inference

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    # ---------------------------------------------------------------- patches
    # ONNX has no complex dtype: real-valued RoPE with a (S, D/2, 2) cos/sin table.
    def precompute_real(dim, end, theta=10000.0, device=None):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        f = torch.outer(torch.arange(end, dtype=torch.float32, device=device), freqs)
        return torch.stack([torch.cos(f), torch.sin(f)], dim=-1)

    def apply_rotary_real(x, freqs):
        xf = x.float().reshape(*x.shape[:3], -1, 2)
        xr, xi = xf[..., 0], xf[..., 1]
        c = freqs[None, :, None, :, 0]
        s = freqs[None, :, None, :, 1]
        return torch.stack([xr * c - xi * s, xr * s + xi * c], dim=-1).reshape_as(x).type_as(x)

    M.apply_rotary_emb = apply_rotary_real

    # Branch-free version of _safe_attention_mask (the original branches on tensor data).
    def safe_attention_mask(x, mask):
        mask = mask.to(dtype=torch.bool)
        has_any = mask.any(dim=1)
        x = x * has_any[:, None, None].to(x.dtype)
        first = mask[:, :1] | ~has_any[:, None]
        return x, torch.cat([first, mask[:, 1:]], dim=1)

    M._safe_attention_mask = safe_attention_mask

    # The original plan builder calls int(query_len), which silently specializes the dynamic
    # frame axis to the example length. Keep it symbolic (FA3 is never used in the export).
    from irodori_tts.attention import ContextAttentionPlan

    def build_plan(segment_masks, *, query_len, attention_dtype):
        kv_mask = segment_masks[0] if len(segment_masks) == 1 else torch.cat(segment_masks, dim=1)
        return ContextAttentionPlan(kv_mask=kv_mask, query_len=query_len, use_fa3=False)

    M.build_context_attention_plan = build_plan

    # The exporter lowers a boolean-masked SDPA to a softmax followed by IsNaN -> Where (zeroing fully masked
    # rows). WebGPU has no IsNaN kernel, so every attention layer copied its score matrix to the CPU and back:
    # ~155 of 276 ms per DiT step (RTX 4050, 438 frames). Irodori's masks always leave at least one valid key
    # per row that matters (self tokens in joint attention, the safe-mask first key, ModernBERT's
    # key | ~any), so an additive mask is equivalent there; rows that are fully masked (padding queries) are
    # masked out downstream anyway.
    _sdpa_orig = torch.nn.functional.scaled_dot_product_attention

    def sdpa_additive(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        if is_causal or dropout_p or enable_gqa:
            return _sdpa_orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                              scale=scale, enable_gqa=enable_gqa)
        scale = query.shape[-1] ** -0.5 if scale is None else scale
        scores = torch.matmul(query, key.transpose(-2, -1)).float() * scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores + (attn_mask.float() - 1.0) * 1e9
            else:
                scores = scores + attn_mask.float()
        return torch.matmul(torch.softmax(scores, dim=-1).to(value.dtype), value)

    torch.nn.functional.scaled_dot_product_attention = sdpa_additive

    # ---------------------------------------------------------------- model
    ckpt = Path(args.checkpoint)
    state, cfgd, infer_cfg, enc_cfg = _load_checkpoint_for_inference(ckpt)
    cfg = merge_dataclass_overrides(ModelConfig(), cfgd, section="m")
    if not cfg.use_pretrained_text_encoder:
        raise SystemExit("Only pretrained text encoder (v4) checkpoints are supported.")
    model = M.TextToLatentRFDiT(cfg, pretrained_backbone_config=enc_cfg, load_pretrained_backbone_weights=False)
    model.load_state_dict(state, assign=True)
    model = model.float().eval()
    del state
    for enc in [model.speaker_encoder] if model.speaker_encoder is not None else []:
        hd = cfg.speaker_dim // cfg.speaker_heads
        enc._rope_freqs = lambda n, device, hd=hd: precompute_real(hd, n, device=device)
    hd_model = cfg.model_dim // cfg.num_heads
    model._rope_freqs = lambda n, device: precompute_real(hd_model, n, device=device)

    # ModernBERT with explicitly built boolean masks (HF mask factories are not exportable).
    backbone = model.pretrained_text_backbone.backbone
    window = int(backbone.config.local_attention) // 2

    # The token embedding lookup (102400 x 768 table, ~315 MB) exceeds WebGPU buffer binding limits,
    # so the graph takes embeddings; the table is saved separately and gathered on the CPU.
    def backbone_forward(inputs_embeds, mask):
        key = mask | ~mask.any(dim=1, keepdim=True)
        full = key[:, None, None, :]
        idx = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        sliding = full & ((idx[:, None] - idx[None, :]).abs() <= window)[None, None]
        h = backbone(inputs_embeds=inputs_embeds, attention_mask={"full_attention": full, "sliding_attention": sliding},
                     return_dict=True).last_hidden_state
        return h * mask.unsqueeze(-1).to(h.dtype)

    model.pretrained_text_backbone.forward = backbone_forward

    def export(module, inputs, names_in, names_out, dyn, name, check=None):
        # --only dacvae_decoder also selects its fp16 twin.
        if args.only and name not in args.only and not (name == "dacvae_decoder_fp16" and "dacvae_decoder" in args.only):
            return None
        t0 = time.time()
        with torch.inference_mode():
            ref = module(*inputs)
        prog = torch.onnx.export(module, inputs, input_names=names_in, output_names=names_out,
                                 dynamic_shapes=dyn, dynamo=True, opset_version=args.opset, optimize=True)
        prog.save(str(out / f"{name}.onnx"), external_data=True)
        # Fail loudly if an axis declared dynamic was specialized to the example size.
        import onnx
        graph = onnx.load(str(out / f"{name}.onnx"), load_external_data=False).graph
        for inp in graph.input:
            for axis in (dyn or {}).get(inp.name) or {}:
                d = inp.type.tensor_type.shape.dim[axis]
                if not d.dim_param:
                    raise SystemExit(f"{name}: input '{inp.name}' axis {axis} was exported as static size {d.dim_value}.")
        # WebGPU has no IsNaN kernel: each one runs on the CPU with a GPU -> CPU -> GPU copy around it.
        n_isnan = sum(1 for n in graph.node if n.op_type == "IsNaN")
        if n_isnan:
            print(f"[warn] {name}.onnx has {n_isnan} IsNaN node(s); they fall back to the CPU on WebGPU.", flush=True)
        n_patched = patch_bool_ops(out / f"{name}.onnx")
        print(f"[export] {name}.onnx ({time.time() - t0:.0f}s, bool-op patches: {n_patched})", flush=True)
        return ref

    D = torch.export.Dim
    B = D("batch", min=1, max=64)

    # ---------------------------------------------------------------- text / caption encoder
    class TextEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, text_embeds, text_mask, caption_embeds, caption_mask):
            m = self.m
            ts = m.text_norm(m.text_encoder(m.pretrained_text_backbone, text_embeds, text_mask))
            cs = m.caption_norm(m.caption_encoder(m.pretrained_text_backbone, caption_embeds, caption_mask))
            return ts, cs

    class TextOnlyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, text_embeds, text_mask):
            m = self.m
            return m.text_norm(m.text_encoder(m.pretrained_text_backbone, text_embeds, text_mask))

    from safetensors.torch import save_file
    table = backbone.embeddings.tok_embeddings.weight.detach().float().contiguous()
    save_file({"tok_embeddings": table}, str(out / "text_embedding.safetensors"))
    lt, lc = 48, 32
    hidden = table.shape[1]
    ids = torch.randn(2, lt, hidden)
    tmask = torch.arange(lt)[None].repeat(2, 1) < torch.tensor([[20], [48]])
    cids = torch.randn(2, lc, hidden)
    cmask = torch.arange(lc)[None].repeat(2, 1) < torch.tensor([[10], [0]])
    T, C = D("text_len", min=2, max=2048), D("cap_len", min=2, max=2048)
    if cfg.use_caption_condition:
        export(TextEncoder().eval(), (ids, tmask, cids, cmask), ["text_embeds", "text_mask", "caption_embeds", "caption_mask"],
               ["text_state", "caption_state"],
               {"text_embeds": {0: B, 1: T}, "text_mask": {0: B, 1: T}, "caption_embeds": {0: B, 1: C}, "caption_mask": {0: B, 1: C}},
               "text_encoder")
    else:
        export(TextOnlyEncoder().eval(), (ids, tmask), ["text_embeds", "text_mask"], ["text_state"],
               {"text_embeds": {0: B, 1: T}, "text_mask": {0: B, 1: T}}, "text_encoder")

    # ---------------------------------------------------------------- speaker encoder
    if cfg.use_speaker_condition_resolved:
        class SpeakerEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model

            def forward(self, ref_latent, ref_mask):  # already speaker-patched in C#
                m = self.m
                s = m.speaker_norm(m.speaker_encoder(ref_latent, ref_mask))
                return m._prepend_masked_mean_token(s, ref_mask)

        P = D("ref_len", min=2, max=4096)
        lat = torch.randn(2, 20, cfg.speaker_patched_latent_dim)
        rmask = torch.arange(20)[None].repeat(2, 1) < torch.tensor([[20], [0]])
        export(SpeakerEncoder().eval(), (lat, rmask), ["ref_latent", "ref_mask"], ["speaker_state", "speaker_mask"],
               {"ref_latent": {0: B, 1: P}, "ref_mask": {0: B, 1: P}}, "speaker_encoder")

    # ---------------------------------------------------------------- duration predictor
    if cfg.use_duration_predictor:
        arch = str(cfg.duration_architecture)
        if arch not in ("token_sum_adarn_zero_no_aux", "token_sum_dual_adarn_zero_no_aux"):
            raise SystemExit(f"Duration architecture {arch} is not supported by the ONNX export.")

        class Duration(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model

            def forward(self, text_state, text_mask, features, speaker_state, has_speaker, caption_state, caption_mask, has_caption):
                return self.m.duration_predictor(text_state, text_mask=text_mask, aux_features=features,
                                                 speaker_state=speaker_state, has_speaker=has_speaker,
                                                 caption_state=caption_state, caption_mask=caption_mask,
                                                 has_caption=has_caption).float()

        S2, C2 = D("spk_len", min=1, max=4096), D("cap_len", min=2, max=2048)
        dur_in = (torch.randn(2, lt, cfg.text_dim), tmask, torch.rand(2, cfg.duration_aux_dim),
                  torch.randn(2, 9, cfg.speaker_dim), torch.tensor([True, False]),
                  torch.randn(2, lc, cfg.caption_dim_resolved), cmask, torch.tensor([True, False]))
        export(Duration().eval(), dur_in,
               ["text_state", "text_mask", "features", "speaker_state", "has_speaker", "caption_state", "caption_mask", "has_caption"],
               ["log_frames"],
               {"text_state": {0: B, 1: T}, "text_mask": {0: B, 1: T}, "features": {0: B}, "speaker_state": {0: B, 1: S2},
                "has_speaker": {0: B}, "caption_state": {0: B, 1: C2}, "caption_mask": {0: B, 1: C2}, "has_caption": {0: B}},
               "duration")

    # ---------------------------------------------------------------- DiT step
    meanflow = str(cfg.flow_parameterization).lower() == "meanflow"

    class DiTStep(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, x_t, t, text_state, text_mask, speaker_state, speaker_mask, caption_state, caption_mask, delta_t=None):
            return self.m.forward_with_encoded_conditions(x_t, t, text_state, text_mask, speaker_state, speaker_mask,
                                                          caption_state, caption_mask, delta_t=delta_t)

    if args.dit_precision == "auto":
        args.dit_precision = "fp32" if meanflow else "fp16"
    dit_dtype = torch.float16 if args.dit_precision == "fp16" else torch.float32
    model.to(dit_dtype)
    Fr, T3, S3, C3 = D("frames", min=2, max=8192), D("text_len", min=2, max=2048), D("spk_len", min=1, max=4096), D("cap_len", min=2, max=2048)
    dit_in = [torch.randn(2, 60, cfg.patched_latent_dim).to(dit_dtype), torch.tensor([0.7, 0.3]).to(torch.float32 if meanflow else dit_dtype),
              torch.randn(2, lt, cfg.text_dim).to(dit_dtype), tmask,
              torch.randn(2, 9, cfg.speaker_dim).to(dit_dtype), torch.ones(2, 9, dtype=torch.bool),
              torch.randn(2, lc, cfg.caption_dim_resolved).to(dit_dtype), cmask]
    names = ["x_t", "t", "text_state", "text_mask", "speaker_state", "speaker_mask", "caption_state", "caption_mask"]
    dyn = {"x_t": {0: B, 1: Fr}, "t": {0: B}, "text_state": {0: B, 1: T3}, "text_mask": {0: B, 1: T3},
           "speaker_state": {0: B, 1: S3}, "speaker_mask": {0: B, 1: S3}, "caption_state": {0: B, 1: C3}, "caption_mask": {0: B, 1: C3}}
    if meanflow:
        dit_in.append(torch.tensor([0.25, 0.25]))
        names.append("delta_t")
        dyn["delta_t"] = {0: B}
    if args.legacy_dit_step:
        export(DiTStep().eval(), tuple(dit_in), names, ["v"], dyn, "dit_step")

    # ---------------------------------------------------------------- DiT context K/V + step over cached K/V
    # The conditioning K/V projections do not depend on x_t or t, so Python computes them once per request
    # (build_context_kv_cache). dit_step.onnx recomputed them in every layer of every step; split them out:
    #   dit_context.onnx : states -> per-layer K/V of the concatenated context (text[, speaker][, caption])
    #   dit_step_kv.onnx : one velocity evaluation that attends over those K/V
    # The runtime keeps the K/V on the GPU between steps (IOBinding).
    att0 = model.blocks[0].attention
    has_spk, has_cap = att0.has_speaker_condition, att0.has_caption_condition
    n_layers = len(model.blocks)

    class DiTContext(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, text_state, speaker_state, caption_state, speaker_kv_scale):
            # speaker_kv_scale: (layers,) multiplier on the speaker K/V of each layer (speaker_kv_scale /
            # speaker_kv_max_layers in Python); ones leave the projections untouched.
            ks, vs = [], []
            for i, blk in enumerate(self.m.blocks):
                p = blk.attention.project_context_kv(text_context=text_state,
                                                     speaker_context=speaker_state if has_spk else None,
                                                     caption_context=caption_state if has_cap else None)
                k_parts, v_parts, o = [p[0]], [p[1]], 2
                if has_spk:
                    s = speaker_kv_scale[i].to(p[o].dtype)
                    k_parts.append(p[o] * s)
                    v_parts.append(p[o + 1] * s)
                    o += 2
                if has_cap:
                    k_parts.append(p[o])
                    v_parts.append(p[o + 1])
                ks.append(torch.cat(k_parts, dim=1))
                vs.append(torch.cat(v_parts, dim=1))
            return torch.stack(ks), torch.stack(vs)  # (layers, B, S_ctx, H, Dh)

    def attention_over_cache(att, x, k_ctx, v_ctx, freqs, plan):
        # JointAttention.forward with the projected context already concatenated in kv order.
        bsz, seq_len, _ = x.shape
        q = att.q_norm(att.wq(x).reshape(bsz, seq_len, att.heads, att.head_dim))
        k_self = att.k_norm(att.wk(x).reshape(bsz, seq_len, att.heads, att.head_dim))
        v_self = att.wv(x).reshape(bsz, seq_len, att.heads, att.head_dim)
        q = att._apply_rotary_half(q, freqs[:seq_len])
        k_self = att._apply_rotary_half(k_self, freqs[:seq_len])
        k = torch.cat([k_self, k_ctx], dim=1)
        v = torch.cat([v_self, v_ctx], dim=1)
        y = M.context_attention(q, k, v, plan).reshape(bsz, seq_len, att.dim)
        return att.wo(y * torch.sigmoid(att.gate(x)))

    class DiTStepKV(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, x_t, t, context_mask, k_context, v_context, delta_t=None):
            m = self.m
            cond = m.cond_module(M.get_timestep_embedding(t, cfg.timestep_embed_dim).to(dtype=x_t.dtype))
            if m.delta_cond_module is not None:
                cond = cond + m.delta_cond_module(M.get_timestep_embedding(delta_t, cfg.timestep_embed_dim).to(dtype=x_t.dtype))
            cond = cond[:, None, :]
            x = m.in_proj(x_t)
            freqs = m._rope_freqs(x.shape[1], x.device)
            self_mask = torch.ones_like(x[:, :, 0], dtype=torch.bool)
            plan = M.ContextAttentionPlan(kv_mask=torch.cat([self_mask, context_mask.to(torch.bool)], dim=1),
                                          query_len=x.shape[1], use_fa3=False)
            for i, blk in enumerate(m.blocks):
                h, gate = blk.attention_adaln(x, cond)
                x = x + gate * attention_over_cache(blk.attention, h, k_context[i], v_context[i], freqs, plan)
                h, gate = blk.mlp_adaln(x, cond)
                x = x + gate * blk.mlp(h)
            return m.out_proj(m.out_norm(x)).to(dtype=x_t.dtype)

    kv_scale = torch.ones(n_layers)
    ctx_in = (dit_in[2], dit_in[4], dit_in[6], kv_scale)
    cmask_all = [dit_in[3]] + ([dit_in[5]] if has_spk else []) + ([dit_in[7]] if has_cap else [])
    with torch.inference_mode():
        kc, vc = DiTContext().eval()(*ctx_in)
        step_in = [dit_in[0], dit_in[1], torch.cat(cmask_all, dim=1), kc, vc] + ([dit_in[8]] if meanflow else [])
    with torch.inference_mode():
        # The split graphs must reproduce the single-step forward.
        ref = DiTStep().eval()(*dit_in) if not args.only or "dit_step_kv" in args.only else None
    if ref is not None:
        with torch.inference_mode():
            got = DiTStepKV().eval()(*step_in)
        err = (ref.float() - got.float()).abs().max().item()
        print(f"[check] dit_step_kv vs forward_with_encoded_conditions: max abs diff {err:.3e}", flush=True)
        if not err < (5e-2 if dit_dtype == torch.float16 else 1e-3):
            raise SystemExit("dit_step_kv does not match the reference forward.")
    Ctx = D("ctx_len", min=2, max=8192)
    export(DiTContext().eval(), ctx_in, ["text_state", "speaker_state", "caption_state", "speaker_kv_scale"],
           ["k_context", "v_context"],
           {"text_state": {0: B, 1: T3}, "speaker_state": {0: B, 1: S3}, "caption_state": {0: B, 1: C3}, "speaker_kv_scale": None},
           "dit_context")
    step_names = ["x_t", "t", "context_mask", "k_context", "v_context"] + (["delta_t"] if meanflow else [])
    step_dyn = {"x_t": {0: B, 1: Fr}, "t": {0: B}, "context_mask": {0: B, 1: Ctx},
                "k_context": {1: B, 2: Ctx}, "v_context": {1: B, 2: Ctx}}
    if meanflow:
        step_dyn["delta_t"] = {0: B}
    export(DiTStepKV().eval(), tuple(step_in), step_names, ["v"], step_dyn, "dit_step_kv")
    model.float()

    # ---------------------------------------------------------------- DACVAE
    import dacvae.nn.layers as L

    def snake_plain(x, alpha):
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        # 1/alpha in fp32 (a constant after export): in fp16, 1e-9 vanishes and small alphas overflow.
        # s * s, not pow(s, 2): WGSL pow is undefined for negative bases, and the WebGPU EP's fp16 Pow broke the
        # audio (the fp32 one happens to work).
        s = torch.sin(alpha * x)
        x = x + (alpha.float() + 1e-9).reciprocal().to(x.dtype) * (s * s)
        return x.reshape(shape)

    L.snake = snake_plain
    from dacvae import DACVAE

    dac = DACVAE.load(args.codec).eval().float()
    for mod in dac.modules():
        if hasattr(mod, "weight_g"):
            torch.nn.utils.remove_weight_norm(mod)

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.d = dac

        def forward(self, latent):  # (B, T, D) -> (B, 1, T * hop), float32 in and out
            dtype = self.d.quantizer.out_proj.weight.dtype
            x = self.d.quantizer.out_proj(latent.transpose(1, 2).to(dtype))
            for layer in self.d.decoder.model:
                x = layer(x)
            pre = self.d.decoder.wm_model.encoder_block.pre  # forward_no_conv: Snake, Conv, Tanh
            return pre[2](pre[1](pre[0](x))).float()

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.d = dac

        def forward(self, wav):  # (B, 1, T) already reflect-padded to a multiple of hop
            z = self.d.encoder(wav)
            mean, _ = self.d.quantizer.in_proj(z).chunk(2, dim=1)
            return mean.transpose(1, 2)

    hop = int(dac.hop_length)
    # ConvTranspose1d(stride s, kernel 2s) as a 2-tap Conv1d with s * C_out channels and an interleave (polyphase
    # form, equal up to float rounding). The WebGPU EP's fp16 ConvTranspose is wrong (the stride-10 upsampler fell
    # to 14 dB against the CPU EP, and the audio with it); Conv is right in fp16 and faster than ConvTranspose.
    class PolyphaseConvTranspose(torch.nn.Module):
        def __init__(self, ct):
            super().__init__()
            s, k = ct.stride[0], ct.kernel_size[0]
            if k != 2 * s or ct.groups != 1 or ct.dilation[0] != 1:
                raise SystemExit(f"unexpected ConvTranspose1d (kernel {k}, stride {s}) in the DACVAE decoder")
            w = ct.weight.detach()  # (C_in, C_out, 2s)
            c_in, c_out, _ = w.shape
            wc = torch.empty(s, c_out, c_in, 2, dtype=w.dtype)
            wc[..., 0] = w[:, :, s:].permute(2, 1, 0)  # tap on x[q - 1]: kernel index r + s
            wc[..., 1] = w[:, :, :s].permute(2, 1, 0)  # tap on x[q]:     kernel index r
            self.weight = torch.nn.Parameter(wc.reshape(s * c_out, c_in, 2))
            self.bias = None if ct.bias is None else torch.nn.Parameter(ct.bias.detach().repeat(s))
            self.s, self.c_out, self.pad, self.out_pad = s, c_out, ct.padding[0], ct.output_padding[0]
            self.unpad = ct.unpad  # NormConvTranspose1d trims after the transposed conv (pad_mode "auto")

        def forward(self, x):
            b, _, n_in = x.shape
            y = torch.nn.functional.conv1d(x, self.weight, self.bias, padding=1)  # (B, s * C_out, n_in + 1)
            y = y.reshape(b, self.s, self.c_out, n_in + 1).permute(0, 2, 3, 1).reshape(b, self.c_out, (n_in + 1) * self.s)
            n_out = (n_in + 1) * self.s - 2 * self.pad + self.out_pad
            return self.unpad(y[:, :, self.pad:self.pad + n_out])

    def to_polyphase(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.ConvTranspose1d):
                setattr(module, name, PolyphaseConvTranspose(child))
            else:
                to_polyphase(child)

    decoder = Decoder().eval()
    with torch.no_grad():
        probe = torch.randn(1, 23, dac.quantizer.codebook_dim)
        before = decoder(probe)
        to_polyphase(decoder)
        print(f"[check] polyphase decoder vs ConvTranspose: max abs diff {(decoder(probe) - before).abs().max().item():.3e}", flush=True)
    dyn_latent = {"latent": {0: B, 1: D("frames", min=1, max=8192)}}
    export(decoder, (torch.randn(1, 40, dac.quantizer.codebook_dim),), ["latent"], ["audio"], dyn_latent, "dacvae_decoder")
    if not args.no_decoder_fp16:
        # Inputs and outputs stay float32. On WebGPU (RTX 4050) a 4.6 s pass takes 0.23 s instead of 0.39 s.
        decoder.half()
        export(decoder, (torch.randn(1, 40, dac.quantizer.codebook_dim),), ["latent"], ["audio"], dyn_latent, "dacvae_decoder_fp16")
        dac.float()
    export(Encoder().eval(), (torch.randn(1, 1, hop * 30) * 0.1,), ["wav"], ["latent"],
           {"wav": {0: B, 2: D("blocks", min=1, max=8192) * hop}}, "dacvae_encoder")

    # ---------------------------------------------------------------- config + tokenizer
    from dataclasses import asdict
    config = {
        "format_version": 1,
        "model": asdict(cfg),
        "inference": infer_cfg or {},
        "dit_precision": args.dit_precision,
        "text_embedding": "text_embedding.safetensors",
        "codec": {"sample_rate": int(dac.sample_rate), "hop_length": hop, "latent_dim": int(dac.quantizer.codebook_dim)},
    }
    (out / "irodori_onnx.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    for cand in (ckpt.parent / "tokenizer", ckpt.parent.parent / "tokenizer"):
        if (cand / "tokenizer.json").is_file():
            shutil.copytree(cand, out / "tokenizer", dirs_exist_ok=True)
            break
    else:
        print("[warn] tokenizer/ not found next to the checkpoint; copy tokenizer.json manually.")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
