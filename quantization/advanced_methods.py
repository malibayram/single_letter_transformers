"""advanced_methods.py — AWQ, SmoothQuant ve GPTQ fikirlerini küçük sayılarda gör.

Bu dosya paper/repository implementasyonlarının yerine geçmez. Her yöntemin
ayırt edici matematik fikrini çalışan küçük bir experiment'e indirger:

  1. AWQ-like scale search:
     activation'ı büyük channel'lara weight quantization sırasında koruma payı.

  2. SmoothQuant transform:
     XW = (X / s)(W * s) eşitliğiyle activation outlier zorluğunu weight'e taşı.

  3. GPTQ toy row:
     calibration Hessian bilgisiyle bir weight'i quantize ettikten sonra
     kalan weight'lerde error compensation yap.

Çalıştır:
    PYTHONPATH=quantization .venv/bin/python quantization/advanced_methods.py
"""

import torch

from core import quantize_tensor


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def quantize_weight_int4(weight: torch.Tensor) -> torch.Tensor:
    """One symmetric group per output row: simple W4 baseline."""
    return quantize_tensor(
        weight,
        bits=4,
        group_size=weight.shape[1],
        symmetric=True,
        scale_dtype=torch.float32,
    ).dequantize()


def quantize_whole_tensor_int8(tensor: torch.Tensor) -> torch.Tensor:
    """One scale for an entire tensor, used to expose activation outliers."""
    flat = tensor.flatten()
    quantized = quantize_tensor(
        flat,
        bits=8,
        group_size=flat.numel(),
        symmetric=True,
        scale_dtype=torch.float32,
    )
    return quantized.dequantize().reshape_as(tensor)


def awq_like_demo() -> None:
    """Activation-aware scaling before weight-only INT4."""
    rule("1. AWQ intuition: important activation channel'larını koru")
    torch.manual_seed(0)
    in_features, out_features = 16, 8
    x_cal = torch.randn(256, in_features)
    x_test = torch.randn(256, in_features)
    # İki input channel diğerlerinden çok daha büyük activation üretsin.
    x_cal[:, 0] *= 10
    x_test[:, 0] *= 10
    x_cal[:, 3] *= 4
    x_test[:, 3] *= 4
    weight = torch.randn(out_features, in_features) * 0.5

    target_cal = x_cal @ weight.T
    target_test = x_test @ weight.T

    naive_weight = quantize_weight_int4(weight)
    naive_cal_mse = (x_cal @ naive_weight.T - target_cal).square().mean().item()
    naive_test_mse = (x_test @ naive_weight.T - target_test).square().mean().item()

    # Calibration activation magnitude tells us which input channels matter.
    importance = x_cal.abs().mean(dim=0).clamp_min(1e-5)
    best = None
    for alpha_step in range(21):
        alpha = alpha_step / 20
        channel_scale = importance.pow(alpha)
        # Keep numbers well-conditioned; the paired transform remains exact.
        channel_scale /= (channel_scale.max() * channel_scale.min()).sqrt()

        # Exact before quantization:
        #     X W^T = (X / s) (W * s)^T
        scaled_x = x_cal / channel_scale
        scaled_weight = weight * channel_scale
        assert torch.allclose(
            target_cal, scaled_x @ scaled_weight.T, atol=2e-5
        )

        candidate = quantize_weight_int4(scaled_weight)
        mse = (scaled_x @ candidate.T - target_cal).square().mean().item()
        if best is None or mse < best["cal_mse"]:
            best = {
                "alpha": alpha,
                "scale": channel_scale,
                "weight": candidate,
                "cal_mse": mse,
            }

    awq_test = (
        (x_test / best["scale"]) @ best["weight"].T - target_test
    ).square().mean().item()

    print(f"  naive round-to-nearest calibration MSE : {naive_cal_mse:.6f}")
    print(f"  best alpha                           : {best['alpha']:.2f}")
    print(f"  activation-aware calibration MSE    : {best['cal_mse']:.6f}")
    print(f"  naive held-out test MSE              : {naive_test_mse:.6f}")
    print(f"  activation-aware held-out test MSE   : {awq_test:.6f}")
    print(
        "\n  Buradaki grid search AWQ'nun ana sezgisini gösterir. Full AWQ;\n"
        "  gerçek transformer layer'ları, clipping/search ayrıntıları ve\n"
        "  hardware-friendly group quantization ile daha kapsamlıdır."
    )
    assert best["cal_mse"] < naive_cal_mse
    assert awq_test < naive_test_mse


def smoothquant_demo() -> None:
    """Move activation dynamic range into weights before W8A8."""
    rule("2. SmoothQuant intuition: X outlier zorluğunu W'ye taşı")
    torch.manual_seed(1)
    in_features, out_features = 16, 8
    activations = torch.randn(256, in_features)
    activations[:, 0] *= 50
    activations[:, 3] *= 10
    weight = torch.randn(out_features, in_features) * 0.5
    target = activations @ weight.T

    naive_xq = quantize_whole_tensor_int8(activations)
    naive_wq = quantize_whole_tensor_int8(weight)
    naive_mse = (naive_xq @ naive_wq.T - target).square().mean().item()

    # SmoothQuant paper's central channel transformation. alpha balances how
    # much range is moved from X to W.
    alpha = 0.5
    x_max = activations.abs().amax(dim=0).clamp_min(1e-5)
    w_max = weight.abs().amax(dim=0).clamp_min(1e-5)
    smooth_scale = x_max.pow(alpha) / w_max.pow(1 - alpha)
    smooth_x = activations / smooth_scale
    smooth_w = weight * smooth_scale

    exact_error = (smooth_x @ smooth_w.T - target).abs().max().item()
    smooth_xq = quantize_whole_tensor_int8(smooth_x)
    smooth_wq = quantize_whole_tensor_int8(smooth_w)
    smooth_mse = (smooth_xq @ smooth_wq.T - target).square().mean().item()

    print(f"  exact transform max error       : {exact_error:.3e}")
    print(f"  naive per-tensor W8A8 MSE       : {naive_mse:.6f}")
    print(f"  smoothed per-tensor W8A8 MSE    : {smooth_mse:.6f}")
    print(
        "\n  Equality quantization'dan önce korunur. Kazanç, X ve W dynamic\n"
        "  range'lerinin artık 8-bit seviyeleri daha verimli kullanmasından gelir."
    )
    assert exact_error < 2e-5
    assert smooth_mse < naive_mse


def gptq_toy_demo() -> None:
    """Sequential second-order error compensation for one weight row."""
    rule("3. GPTQ intuition: bir hatayı sonraki weight'lerle telafi et")
    torch.manual_seed(14)
    n_weights, n_samples = 16, 128
    weight = torch.randn(n_weights) * 0.7
    calibration_x = torch.randn(n_weights, n_samples)
    calibration_x[0] *= 8
    calibration_x[3] *= 4

    scale = weight.abs().max() / 7

    def quantize_value(value):
        return (value / scale).round().clamp(-7, 7) * scale

    naive = quantize_value(weight)

    # H approximates how calibration inputs weight different directions.
    hessian = 2 * (calibration_x @ calibration_x.T) / n_samples
    damp = 0.01 * torch.diag(hessian).mean()
    hessian += damp * torch.eye(n_weights)

    # The compact GPTQ update uses an upper Cholesky factor of H^-1.
    h_inv_factor = torch.linalg.cholesky(
        torch.linalg.inv(hessian), upper=True
    )

    working = weight.clone()
    compensated = torch.zeros_like(weight)
    for index in range(n_weights):
        q = quantize_value(working[index])
        compensated[index] = q
        error = (working[index] - q) / h_inv_factor[index, index]
        # Push the current quantization error into remaining coordinates in a
        # way shaped by calibration correlations.
        working[index:] -= error * h_inv_factor[index, index:]

    target = weight @ calibration_x
    naive_mse = (naive @ calibration_x - target).square().mean().item()
    gptq_mse = (
        compensated @ calibration_x - target
    ).square().mean().item()

    changed_codes = int((naive != compensated).sum().item())
    print(f"  fixed INT4 scale                 : {scale.item():.6f}")
    print(f"  codes changed by compensation    : {changed_codes}/{n_weights}")
    print(f"  naive calibration output MSE     : {naive_mse:.6f}")
    print(f"  GPTQ-like compensated output MSE : {gptq_mse:.6f}")
    print(
        "\n  Bu tek-row experiment GPTQ'nun error propagation fikridir. Full GPTQ\n"
        "  layer blokları, group-wise quantizer, damping/permutation ve hızlı\n"
        "  kernel/storage entegrasyonu içerir."
    )
    assert changed_codes > 0
    assert gptq_mse < naive_mse


def main() -> None:
    awq_like_demo()
    smoothquant_demo()
    gptq_toy_demo()
    print("\nBütün advanced-method experiment'leri geçti.\n")


if __name__ == "__main__":
    main()
