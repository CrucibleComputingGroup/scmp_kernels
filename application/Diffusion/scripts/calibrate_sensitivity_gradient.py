#!/usr/bin/env python3
"""
Gradient-based end-to-end sensitivity calibration (Method 3).

For each (op, layer_bucket, timestep_bucket), estimate the squared Jacobian
norm  ‖∂eps_pred / ∂(matmul_output)‖²  via the Hutchinson trace estimator
(K random projections), where eps_pred is the noise prediction at the
chosen timestep.  The output is a sensitivity table that downstream
threshold calibration can use as a per-bucket weight, replacing the
"per-op MSE" proxy with a downstream-image-aware signal.

Run mode: FP-only.  All SC operators are disabled at the controller, so
the model runs as a plain quantized W8A8 teacher and gradients flow
through standard autograd.  No SC kernels are invoked, so no source
modifications are required — instrumentation is external (forward hooks
on Linear modules + per-instance __class__ swap on SCAttention to capture
QK / AV intermediates).  Original logic is restored on exit.

Outputs JSON:
    sensitivity_raw[op:tT:lL]            -- mean grad-norm² per bucket
    sensitivity_per_op_normalized[...]   -- normalized within each op
    selected_timesteps, num_projections, num_calib_samples, ...

Example:
    python scripts/calibrate_sensitivity_gradient.py \\
        --model DiT-XL/2 --image_size 256 --num_sampling_steps 250 \\
        --ckpt /path/to/DiT-XL-2-256x256.pt \\
        --num_calib_samples 8 --num_projections 12 \\
        --timestep_buckets 4 --layer_buckets 4 \\
        --sens_output_json sensitivity_grad.json \\
        --sens_per_block_json sensitivity_grad_per_block.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import seed_everything

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusion import create_diffusion
from models.models import DiT_models
from qdit.sc_integration import (
    OPERATORS,
    add_sc_wrapper,
    create_sc_controller_from_args,
    quantize_sc_model,
)
from qdit.qLinearLayer import QLinearLayer
from qdit.quant import Quantizer
from qdit.sc_integration.sc_attention import SCAttention
from qdit.sc_integration.sc_mlp import SCMlp
from quant_sc_main import create_argparser
from utils.download import find_model


# Forward populates this; backward reads it.  Cleared each Hutchinson iteration.
# Each entry stores  (captured_tensor, sigma_factor) , where
#   sigma_factor = K_contraction · max|operand_a|² · max|operand_b|²
# is the SC noise-variance proxy for this op's output at unit stoc_len:
#   E[‖SC_out − FP_out‖²]  ≈  σ²(L) · ‖output‖_size, with σ²(L) ∝ sigma_factor / L.
# Combined with the gradient norm we get the end-to-end bucket weight
#   bucket_weight ≈ ‖J‖²_F · sigma_factor / L  (ranking is L-independent).
_CAPTURE: dict[int, dict[str, tuple[torch.Tensor, float]]] = defaultdict(dict)


# Module-global, set by main() from --range_quantile (default 1.0 = strict max).
# Strict max best matches empirical fc2-top observation; lower quantiles
# (e.g. 0.999) clip outlier blocks at the cost of muting the σ² signal.
_RANGE_QUANTILE: float = 1.0


def _operand_range(t: torch.Tensor) -> float:
    """Range proxy used inside _sigma_factor.  Returns max|t| when
    _RANGE_QUANTILE == 1.0; otherwise the q-quantile of |t|.

    Switching off strict max (q < 1) gives an outlier-robust proxy that
    approximates Q-DiT's group-wise quantization (which clips per-group
    rather than per-tensor).  Cheaper but mutes the signal at outlier-heavy
    blocks (e.g. block 2's AdaLN-modulated input).
    """
    flat = t.detach().float().abs().reshape(-1)
    if _RANGE_QUANTILE >= 1.0:
        return float(flat.max().item())
    if flat.numel() > 1_000_000:
        idx = torch.randint(0, flat.numel(), (1_000_000,), device=flat.device)
        flat = flat[idx]
    return float(flat.quantile(_RANGE_QUANTILE).item())


def _sigma_factor(operand_a: torch.Tensor, operand_b: torch.Tensor, K: int) -> float:
    """SC noise-variance prefactor for matmul  out = a @ b  (contraction K).

        σ²_per_output_elem ∝ K · range_a² · range_b² / L   (unit-L prefactor)
    """
    range_a_sq = _operand_range(operand_a) ** 2
    range_b_sq = _operand_range(operand_b) ** 2
    return float(K) * range_a_sq * range_b_sq


# ---------------------------------------------------------------------------
# Instrumented forward — verbatim copy of SCAttention.forward FP-only branches,
# with retain_grad on QK pre-softmax and AV matmul outputs.  Active only via
# __class__ swap during calibration; restored to SCAttention afterwards.
# ---------------------------------------------------------------------------
def _instrumented_attn_forward(self, x):
    B, N, C = x.shape

    if self.reorder_index_qkv is not None:
        x = torch.index_select(x, 2, self.reorder_index_qkv)
    x = self.input_quant(x)

    qkv = self.qkv(x)  # input_proj — captured externally via Linear forward_hook

    qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    if self.quantize_bmm_input:
        q = self.q_quant(q)
        k = self.k_quant(k)
        v = self.v_quant(v)

    q_scaled = q * self.scale
    sigma_qk = _sigma_factor(q_scaled, k, K=q_scaled.shape[-1])
    attn_logits = q_scaled @ k.transpose(-2, -1)
    if attn_logits.requires_grad:
        attn_logits.retain_grad()
    _CAPTURE[self.block_idx]["qk"] = (attn_logits, sigma_qk)

    attn = attn_logits.softmax(dim=-1)
    attn = self.attn_drop(attn)

    sigma_av = _sigma_factor(attn, v, K=attn.shape[-1])
    av_out = attn @ v
    if av_out.requires_grad:
        av_out.retain_grad()
    _CAPTURE[self.block_idx]["av"] = (av_out, sigma_av)

    x = av_out.transpose(1, 2).reshape(B, N, C)
    if self.reorder_index_proj is not None:
        x = torch.index_select(x, 2, self.reorder_index_proj)
    x = self.act_quant(x)

    x = self.proj(x)  # proj — captured externally via Linear forward_hook
    x = self.proj_drop(x)
    return x


class _InstrumentedSCAttention(SCAttention):
    forward = _instrumented_attn_forward


def _make_linear_hook(block_idx: int, op_name: str):
    """Forward hook for QLinearLayer / nn.Linear: retains output grad and
    records the SC noise-variance prefactor for the matmul (K · range_x² · range_w²)."""
    def hook(module, inputs, output):
        x = inputs[0]
        if output.requires_grad:
            output.retain_grad()
        K = int(x.shape[-1])
        sigma = _sigma_factor(x, module.weight, K=K)
        _CAPTURE[block_idx][op_name] = (output, sigma)
    return hook


def _install_ste_on_quantizer(cleanups):
    """Monkey-patch Quantizer.forward with a straight-through estimator so
    activation quantization stops severing the autograd graph.  Forward
    values are identical (same W8A8 quant); backward acts as identity.
    Restored on exit via the cleanup list.

    Original forward is `@torch.no_grad()`-decorated, which detaches its
    output.  STE wrapper:  out = x + (x_q - x).detach()  — numerically equal
    to x_q in forward, gradient flows through x in backward.
    """
    original_forward = Quantizer.forward

    def _ste_forward(self, x):
        x_q = original_forward(self, x)
        if x_q.dtype != x.dtype:
            x_q = x_q.to(dtype=x.dtype)
        return x + (x_q - x).detach()

    Quantizer.forward = _ste_forward
    cleanups.append(lambda: setattr(Quantizer, "forward", original_forward))


def _install_grad_through_qlinear(cleanups):
    """Strip @torch.no_grad() from QLinearLayer.forward for the duration of
    calibration so gradient propagates through every Linear in the model.

    The original forward is `@torch.no_grad()`-decorated and detaches its
    output, which kills the autograd chain at every quantized Linear.  The
    replacement re-uses the same F.linear computation but without the
    no_grad context.  All QLinearLayer parameters are registered as buffers
    (not Parameters) so no spurious param-grad memory is allocated.
    """
    import torch.nn.functional as F

    original_forward = QLinearLayer.forward

    def _grad_forward(self, x):
        return F.linear(x, self.weight, self.bias)

    QLinearLayer.forward = _grad_forward
    cleanups.append(lambda: setattr(QLinearLayer, "forward", original_forward))


def _install_instrumentation(model):
    """Attach external instrumentation; return a list of cleanup callables."""
    cleanups = []
    _install_ste_on_quantizer(cleanups)
    _install_grad_through_qlinear(cleanups)
    for block_idx, block in enumerate(model.blocks):
        attn = block.attn
        mlp = block.mlp
        if not isinstance(attn, SCAttention):
            raise RuntimeError(
                f"block {block_idx}.attn is {type(attn).__name__}, expected SCAttention. "
                "Run add_sc_wrapper before installing instrumentation."
            )
        if not isinstance(mlp, SCMlp):
            raise RuntimeError(
                f"block {block_idx}.mlp is {type(mlp).__name__}, expected SCMlp."
            )

        h1 = attn.qkv.register_forward_hook(_make_linear_hook(block_idx, "input_proj"))
        h2 = attn.proj.register_forward_hook(_make_linear_hook(block_idx, "proj"))
        h3 = mlp.fc1.register_forward_hook(_make_linear_hook(block_idx, "mlp_fc1"))
        h4 = mlp.fc2.register_forward_hook(_make_linear_hook(block_idx, "mlp_fc2"))
        for h in (h1, h2, h3, h4):
            cleanups.append(h.remove)

        original_cls = attn.__class__
        attn.__class__ = _InstrumentedSCAttention
        cleanups.append(lambda a=attn, c=original_cls: setattr(a, "__class__", c))

    return cleanups


def _disable_sc(controller):
    """Make every operator in every block fall through to the FP path."""
    for block_idx in range(controller.total_blocks):
        for op in OPERATORS:
            controller.precision_map.set(block_idx, op, enabled=False, timewise=0.0)
    controller.mp_config = None
    controller.adaptive_mp_config = None
    controller.range_mp_config = None


def _bucket_index(value: int, total: int, num_buckets: int) -> int:
    if num_buckets <= 1 or total <= 1:
        return 0
    ratio = value / max(total - 1, 1)
    return min(num_buckets - 1, int(ratio * num_buckets))


def _select_timesteps(total_timesteps: int, num_buckets: int) -> list[int]:
    """Pick one representative timestep per bucket (mid of bucket).
    Returned in noisy → clean order to match diffusion convention."""
    out = []
    bucket_size = total_timesteps / num_buckets
    for b in range(num_buckets):
        t = int((b + 0.5) * bucket_size)
        t = min(max(t, 0), total_timesteps - 1)
        out.append(t)
    return sorted(set(out), reverse=True)


def _build_parser():
    parser = create_argparser()
    parser.description = "Gradient-based end-to-end sensitivity calibration."
    parser.add_argument(
        "--sens_output_json", type=str,
        default="sensitivity_calibration.json",
        help="Path to write the bucketed sensitivity table.",
    )
    parser.add_argument(
        "--sens_per_block_json", type=str, default=None,
        help="Optional path to also dump un-bucketed per-block sensitivities.",
    )
    parser.add_argument(
        "--num_projections", type=int, default=12,
        help="K for Hutchinson trace estimator (random projections per sample).",
    )
    parser.add_argument(
        "--num_calib_samples", type=int, default=8,
        help="Independent (noise, class) samples per timestep.",
    )
    parser.add_argument(
        "--timestep_buckets", type=int, default=4,
        help="Number of timestep buckets in the output table.",
    )
    parser.add_argument(
        "--layer_buckets", type=int, default=4,
        help="Number of layer buckets in the output table.",
    )
    parser.add_argument(
        "--sens_calib_batch_size", type=int, default=4,
        help="Mini-batch size during forward+backward.",
    )
    parser.add_argument(
        "--use_q_sample", action="store_true",
        help="Build x_t via diffusion.q_sample on Gaussian x_0 instead of "
             "feeding Gaussian noise directly.  Slightly more faithful to the "
             "actual operating distribution at each timestep.",
    )
    parser.add_argument(
        "--range_quantile", type=float, default=1.0,
        help="Quantile used as the operand-range proxy in σ² estimation. "
             "1.0 = strict max(|·|) (default; matches single-scale SC noise model). "
             "0.999 = p99.9, outlier-robust (approximates Q-DiT group quantization).",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this calibration.")
    seed_everything(args.seed)
    device = "cuda"

    global _RANGE_QUANTILE
    _RANGE_QUANTILE = float(args.range_quantile)
    print(f"σ² operand-range proxy: "
          f"{'max(|·|)' if _RANGE_QUANTILE >= 1.0 else f'p{_RANGE_QUANTILE*100:.2f}'}")

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size, num_classes=args.num_classes,
    ).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))

    args.weight_group_size = eval(args.weight_group_size)
    args.act_group_size = eval(args.act_group_size)
    if isinstance(args.weight_group_size, int):
        args.weight_group_size = [args.weight_group_size] * len(model.blocks)
    if isinstance(args.act_group_size, int):
        args.act_group_size = [args.act_group_size] * len(model.blocks)

    sc_controller = create_sc_controller_from_args(args, model)
    _disable_sc(sc_controller)

    print("Adding SC wrappers (no SC kernels will fire — controller fully disabled)...")
    scales = defaultdict(lambda: None)
    model = add_sc_wrapper(model, device, args, scales, sc_controller)
    print("Quantizing wrapped model (W8A8 teacher)...")
    model = quantize_sc_model(model, device, args, sc_controller=sc_controller)

    # Use fp32 for stable backward; freeze parameters (we only need activation grads).
    model.to(device).float().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    timesteps = _select_timesteps(diffusion.num_timesteps, args.timestep_buckets)
    print(f"Selected timesteps: {timesteps}")
    print(f"Operators: {sorted(OPERATORS)}")
    print(f"Layer buckets: {args.layer_buckets}, Timestep buckets: {args.timestep_buckets}")

    cleanups = _install_instrumentation(model)
    accum: dict[tuple[str, int, int], dict[str, float]] = defaultdict(
        lambda: {
            "sum_grad_sq": 0.0,       # ‖J‖²_F   (raw Frobenius)
            "sum_per_elem": 0.0,      # ‖J‖²_F / numel(y)  (per-element J²)
            "sum_sc_weighted": 0.0,   # ‖J‖²_F · sigma_factor  (= bucket weight)
            "sum_sigma_factor": 0.0,  # for diagnostics
            "count": 0,
        }
    )

    rng = torch.Generator(device=device).manual_seed(args.seed)
    n_samples = args.num_calib_samples
    bs = args.sens_calib_batch_size
    num_batches = (n_samples + bs - 1) // bs

    try:
        for t_value in timesteps:
            print(f"\n[t={t_value}] running {n_samples} samples × K={args.num_projections}")
            for batch_i in range(num_batches):
                this_bs = min(bs, n_samples - batch_i * bs)
                if this_bs <= 0:
                    continue

                t_tensor = torch.tensor(
                    [t_value] * this_bs, device=device, dtype=torch.long,
                )
                if args.use_q_sample:
                    x0 = torch.randn(
                        this_bs, 4, latent_size, latent_size,
                        device=device, generator=rng, dtype=torch.float32,
                    )
                    x_t = diffusion.q_sample(x0, t_tensor)
                else:
                    x_t = torch.randn(
                        this_bs, 4, latent_size, latent_size,
                        device=device, generator=rng, dtype=torch.float32,
                    )

                y = torch.randint(
                    0, args.num_classes, (this_bs,),
                    device=device, generator=rng,
                )
                sc_controller.set_timestep(t_value)

                for k_idx in range(args.num_projections):
                    _CAPTURE.clear()
                    x_t_g = x_t.detach().clone().requires_grad_(True)
                    pred = model(x_t_g, t_tensor, y)
                    # DiT with learn_sigma=True returns 8 channels (eps + sigma);
                    # we only want sensitivity wrt the noise-prediction channels.
                    eps_pred = pred[:, :4] if pred.shape[1] == 8 else pred

                    # Hutchinson: ⟨v, eps_pred⟩ with unit-variance v gives an
                    # unbiased estimator of trace(J^T J) when summed over many v.
                    v = torch.randn_like(eps_pred)
                    v = v / (eps_pred.numel() ** 0.5)
                    loss = (v * eps_pred).sum()
                    loss.backward()

                    for block_idx, ops in _CAPTURE.items():
                        for op_name, captured in ops.items():
                            tensor, sigma = captured
                            if tensor.grad is None:
                                continue
                            g_sq = tensor.grad.detach().float().pow(2).sum().item()
                            n_elem = float(tensor.numel())
                            key = (op_name, block_idx, t_value)
                            accum[key]["sum_grad_sq"] += g_sq
                            accum[key]["sum_per_elem"] += g_sq / max(n_elem, 1.0)
                            accum[key]["sum_sc_weighted"] += g_sq * sigma
                            accum[key]["sum_sigma_factor"] += sigma
                            accum[key]["count"] += 1

                    # Drop refs / grads so memory doesn't accumulate across K.
                    for ops in _CAPTURE.values():
                        for captured in ops.values():
                            tensor, _ = captured
                            if tensor.grad is not None:
                                tensor.grad = None
                print(f"  batch {batch_i + 1}/{num_batches} done")
    finally:
        for cleanup in cleanups:
            cleanup()

    # ---- Aggregate per-(op, block, t) into (op, l_bucket, t_bucket) ----
    total_blocks = len(model.blocks)
    total_t = diffusion.num_timesteps

    # Three views:
    #   (a) Frobenius total ‖J‖²_F        — pure local Jacobian
    #   (b) Per-element ‖J‖²_F / numel    — fair cross-op comparison
    #   (c) SC-weighted ‖J‖²_F · σ²       — end-to-end bucket weight (recommended)
    bucketed_total: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    bucketed_per_elem: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    bucketed_sc_weighted: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    bucketed_sigma: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    per_block_total: dict[tuple[str, int, int], float] = {}
    per_block_per_elem: dict[tuple[str, int, int], float] = {}
    per_block_sc_weighted: dict[tuple[str, int, int], float] = {}
    for (op_name, block_idx, t_value), info in accum.items():
        if info["count"] == 0:
            continue
        sens_total = info["sum_grad_sq"] / info["count"]
        sens_per_elem = info["sum_per_elem"] / info["count"]
        sens_sc = info["sum_sc_weighted"] / info["count"]
        sigma = info["sum_sigma_factor"] / info["count"]
        per_block_total[(op_name, block_idx, t_value)] = sens_total
        per_block_per_elem[(op_name, block_idx, t_value)] = sens_per_elem
        per_block_sc_weighted[(op_name, block_idx, t_value)] = sens_sc
        l_b = _bucket_index(block_idx, total_blocks, args.layer_buckets)
        t_b = _bucket_index(t_value, total_t, args.timestep_buckets)
        bucketed_total[(op_name, t_b, l_b)].append(sens_total)
        bucketed_per_elem[(op_name, t_b, l_b)].append(sens_per_elem)
        bucketed_sc_weighted[(op_name, t_b, l_b)].append(sens_sc)
        bucketed_sigma[(op_name, t_b, l_b)].append(sigma)

    def _bucket_to_table(buckets):
        out: dict[str, float] = {}
        for (op_name, t_b, l_b), values in buckets.items():
            out[f"{op_name}:t{t_b}:l{l_b}"] = float(np.mean(values))
        return out

    sensitivity_total = _bucket_to_table(bucketed_total)
    sensitivity_per_elem = _bucket_to_table(bucketed_per_elem)
    sensitivity_sc_weighted = _bucket_to_table(bucketed_sc_weighted)
    sigma_per_bucket = _bucket_to_table(bucketed_sigma)

    def _normalize_within_op(table):
        op_totals: dict[str, float] = defaultdict(float)
        for key, val in table.items():
            op_totals[key.split(":")[0]] += val
        out = {}
        for key, val in table.items():
            op = key.split(":")[0]
            denom = op_totals[op] if op_totals[op] > 0 else 1.0
            out[key] = val / denom
        return out, op_totals

    sens_total_norm, op_totals_total = _normalize_within_op(sensitivity_total)
    sens_per_elem_norm, op_totals_per_elem = _normalize_within_op(sensitivity_per_elem)
    sens_sc_norm, op_totals_sc = _normalize_within_op(sensitivity_sc_weighted)

    op_share_total = {
        op: op_totals_total[op] / (sum(op_totals_total.values()) or 1.0)
        for op in op_totals_total
    }
    op_share_per_elem = {
        op: op_totals_per_elem[op] / (sum(op_totals_per_elem.values()) or 1.0)
        for op in op_totals_per_elem
    }
    op_share_sc_weighted = {
        op: op_totals_sc[op] / (sum(op_totals_sc.values()) or 1.0)
        for op in op_totals_sc
    }

    payload = {
        "method": "gradient_hutchinson",
        "stoc_len_levels": [int(x) for x in args.mp_levels.split(",")],
        "timestep_buckets": args.timestep_buckets,
        "layer_buckets": args.layer_buckets,
        "selected_timesteps": timesteps,
        "num_projections": args.num_projections,
        "num_calib_samples": n_samples,
        "use_q_sample": bool(args.use_q_sample),
        "range_quantile": _RANGE_QUANTILE,
        "operators": sorted({k.split(":")[0] for k in sensitivity_total}),
        # SC-weighted (RECOMMENDED): J²·σ² ≈ end-to-end bucket weight under SC.
        # σ² ∝ K · max|a|² · max|b|², L-independent prefactor.
        "op_share_sc_weighted": op_share_sc_weighted,
        "sensitivity_sc_weighted_raw": sensitivity_sc_weighted,
        "sensitivity_sc_weighted_per_op_normalized": sens_sc_norm,
        "sigma_factor_per_bucket": sigma_per_bucket,
        # Pure Jacobian views (kept for diagnostics):
        "op_share_total": op_share_total,
        "sensitivity_total_raw": sensitivity_total,
        "sensitivity_total_per_op_normalized": sens_total_norm,
        "op_share_per_elem": op_share_per_elem,
        "sensitivity_per_elem_raw": sensitivity_per_elem,
        "sensitivity_per_elem_per_op_normalized": sens_per_elem_norm,
    }

    out_path = Path(args.sens_output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote sensitivity table to {out_path}")

    if args.sens_per_block_json:
        per_block_payload = {
            "total":       {f"{op}:t{t}:b{b}": v for (op, b, t), v in per_block_total.items()},
            "per_elem":    {f"{op}:t{t}:b{b}": v for (op, b, t), v in per_block_per_elem.items()},
            "sc_weighted": {f"{op}:t{t}:b{b}": v for (op, b, t), v in per_block_sc_weighted.items()},
        }
        per_block_path = Path(args.sens_per_block_json)
        per_block_path.parent.mkdir(parents=True, exist_ok=True)
        with open(per_block_path, "w") as f:
            json.dump(per_block_payload, f, indent=2)
        print(f"Wrote per-block sensitivities to {per_block_path}")

    print("\n--- Op-level share, SC-weighted J²·σ² (RECOMMENDED, sums to 1.0) ---")
    for op in sorted(op_share_sc_weighted):
        print(f"  {op:<12s}  {op_share_sc_weighted[op]:.4f}")
    print("\n--- Op-level share, Frobenius J² alone (sums to 1.0) ---")
    for op in sorted(op_share_total):
        print(f"  {op:<12s}  {op_share_total[op]:.4f}")
    print("\n--- Op-level share, per-element J² (sums to 1.0) ---")
    for op in sorted(op_share_per_elem):
        print(f"  {op:<12s}  {op_share_per_elem[op]:.4f}")


if __name__ == "__main__":
    main()
