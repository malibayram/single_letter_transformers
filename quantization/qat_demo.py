"""qat_demo.py — PTQ ile QAT arasındaki farkı çalışan küçük bir görevde gör.

Görev: 12 özellikli sentetik örnekleri iki sınıfa ayıran tek Linear layer.

Akış:
  1. Float model normal şekilde eğitilir.
  2. PTQ: eğitim bittikten sonra doğrudan packed INT4'e çevrilir.
  3. QAT: aynı float model, forward'da INT4 hatasını görerek kısa süre eğitilir.
  4. QAT modeli de gerçek packed INT4 `QuantizedLinear` katmanına çevrilir.

Bu küçük deneyde QAT genellikle PTQ'nun kaybettiği doğruluğun bir bölümünü geri
alır. QAT'nin her görevde iyileşme garantisi olmadığı özellikle unutulmamalı;
calibration, optimizer, learning rate ve quantization config sonucu etkiler.

Çalıştır:
    PYTHONPATH=quantization .venv/bin/python quantization/qat_demo.py
"""

import torch
from torch import nn
import torch.nn.functional as F

from linear import QATLinear, QuantizedLinear


SEED = 0
IN_FEATURES = 12
TRAIN_SAMPLES = 3000
TEST_SAMPLES = 1000
FLOAT_STEPS = 500
QAT_STEPS = 250


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Binary logits > 0 ise sınıf 1."""
    with torch.no_grad():
        predicted = model(x) > 0
        return (predicted == y.bool()).float().mean().item()


def make_data():
    """Sabit bir teacher boundary'den deterministic train/test verisi üret."""
    torch.manual_seed(SEED)
    teacher_weight = torch.randn(1, IN_FEATURES)
    teacher_bias = torch.tensor([0.3])

    x_train = torch.randn(TRAIN_SAMPLES, IN_FEATURES)
    y_train = (x_train @ teacher_weight.T + teacher_bias > 0).float()
    x_test = torch.randn(TEST_SAMPLES, IN_FEATURES)
    y_test = (x_test @ teacher_weight.T + teacher_bias > 0).float()
    return x_train, y_train, x_test, y_test


def train_float(x_train, y_train) -> nn.Linear:
    """Önce normal float model eğit."""
    model = nn.Linear(IN_FEATURES, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
    for _ in range(FLOAT_STEPS):
        indices = torch.randint(TRAIN_SAMPLES, (128,))
        loss = F.binary_cross_entropy_with_logits(
            model(x_train[indices]), y_train[indices]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def main() -> None:
    x_train, y_train, x_test, y_test = make_data()
    float_model = train_float(x_train, y_train)
    float_accuracy = accuracy(float_model, x_test, y_test)

    # PTQ: Float training tamamlandı; weight'e bir daha öğrenme fırsatı vermeden
    # doğrudan packed INT4 storage'a geçiyoruz.
    ptq_model = QuantizedLinear.from_float(
        float_model,
        bits=4,
        group_size=IN_FEATURES,
        symmetric=True,
    )
    ptq_accuracy = accuracy(ptq_model, x_test, y_test)

    # QAT: Aynı float başlangıcı al. QATLinear FP master weight saklar, fakat
    # forward her adımda fake INT4 weight kullanır. STE gradient'i master
    # weight'e taşır.
    qat_model = QATLinear.from_float(
        float_model,
        bits=4,
        group_size=IN_FEATURES,
        symmetric=True,
    )
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=3e-4)
    qat_model.train()
    for step in range(1, QAT_STEPS + 1):
        indices = torch.randint(TRAIN_SAMPLES, (128,))
        logits = qat_model(x_train[indices])
        loss = F.binary_cross_entropy_with_logits(logits, y_train[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in (1, 50, 100, 250):
            current = accuracy(qat_model, x_test, y_test)
            print(f"QAT step {step:3d}: loss={loss.item():.4f}, fake-INT4 acc={current:.1%}")
    qat_model.eval()

    # Convert: FP master weight atılır; geriye qweight + scale (+ bias) kalır.
    qat_packed = qat_model.to_quantized()
    qat_accuracy = accuracy(qat_packed, x_test, y_test)

    print("\nFinal comparison")
    print(f"  Float model          : {float_accuracy:.1%}")
    print(f"  Direct PTQ INT4      : {ptq_accuracy:.1%}")
    print(f"  QAT -> packed INT4   : {qat_accuracy:.1%}")
    print(
        f"\nPacked weight: {qat_packed.qweight.numel()} byte payload + "
        f"{qat_packed.scales.numel()*qat_packed.scales.element_size()} byte scale; "
        f"float weight would be {float_model.weight.numel()*4} byte."
    )
    print(
        "\nNot: QuantizedLinear storage gerçekten packed INT4'tür. Bu eğitim\n"
        "örneğinin forward'u portability için weight'i açıp float matmul yapar;\n"
        "optimized INT4 kernel veya speedup iddiası yoktur."
    )

    assert isinstance(qat_packed, QuantizedLinear)
    assert qat_packed.qweight.dtype == torch.uint8
    assert qat_packed.qweight.numel() * 2 == float_model.weight.numel()
    assert float_accuracy > 0.97
    assert ptq_accuracy > 0.94
    assert qat_accuracy > 0.94


if __name__ == "__main__":
    main()
