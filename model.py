"""
PHASE 5 (Corrected): BiGRU Model for ISLR101
مدل BiGRU برای تشخیص کلمات زبان اشاره ایرانی

قوانین رعایت‌شده:
- ورودی: (B, 3, 112, 67)
  3   = channels
  112 = frames (Temporal / Sequence Length)
  67  = keypoints
- برای GRU تبدیل می‌شود به: (B, 112, 201)   جایی که 201 = 3 × 67
- Sequence Length = 112 (نه 67)
"""

import torch
import torch.nn as nn


class BiGRUModel(nn.Module):
    """
    BiGRU برای طبقه‌بندی زبان اشاره

    ورودی مورد انتظار: (batch, 3, 112, 67)
    خروجی: (batch, num_classes)
    """

    def __init__(
        self,
        input_size: int = 201,          # 3 * 67
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 101,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(gru_output_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, T, 67)   معمولاً T=112

        Returns:
            logits: (B, num_classes)
        """
        # اطمینان از shape پایه
        assert x.ndim == 4 and x.shape[1] == 3 and x.shape[3] == 67, (
            f"Expected input shape (B, 3, T, 67), got {x.shape}"
        )

        batch_size = x.size(0)
        T = x.size(2)  # معمولاً 112

        # (B, 3, T, 67) → (B, T, 3, 67) → (B, T, 201)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, T, self.input_size)

        # GRU
        gru_out, _ = self.gru(x)  # (B, T, hidden*2 or hidden)

        # آخرین timestep
        last_out = gru_out[:, -1, :]  # (B, hidden*2 or hidden)

        last_out = self.dropout(last_out)
        logits = self.fc(last_out)    # (B, num_classes)

        return logits


def count_parameters(model: nn.Module) -> int:
    """شمارش پارامترهای قابل آموزش"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """تست مدل با shape صحیح"""
    print("=" * 60)
    print("PHASE 5 (Corrected): BiGRU Model Testing")
    print("=" * 60)

    config = {
        "input_size": 3 * 67,       # 201
        "hidden_size": 256,
        "num_layers": 2,
        "num_classes": 101,
        "dropout": 0.3,
        "bidirectional": True,
    }

    model = BiGRUModel(**config)

    print(f"\n📐 Model Architecture:")
    print(f"  Expected input : (B, 3, 112, 67)")
    print(f"  After reshape  : (B, 112, {config['input_size']})")
    print(f"  Hidden size    : {config['hidden_size']}")
    print(f"  Num layers     : {config['num_layers']}")
    print(f"  Bidirectional  : {config['bidirectional']}")
    print(f"  Dropout        : {config['dropout']}")
    print(f"  Output         : (B, {config['num_classes']})")

    total_params = count_parameters(model)
    print(f"\n📊 Parameters: {total_params:,}")
    print(f"  Size (float32): {total_params * 4 / (1024 ** 2):.2f} MB")

    # Forward test
    print(f"\n🧪 Testing forward pass:")
    batch_size = 8
    dummy = torch.randn(batch_size, 3, 112, 67)

    model.eval()
    with torch.no_grad():
        out = model(dummy)

    print(f"  Input  shape : {dummy.shape}")
    print(f"  Output shape : {out.shape}")
    print(f"  Output range : [{out.min().item():.4f}, {out.max().item():.4f}]")

    assert out.shape == (batch_size, 101), f"Wrong output shape: {out.shape}"
    print("  ✅ Output shape correct")

    # GPU test
    if torch.cuda.is_available():
        print(f"\n🖥️  Testing on GPU:")
        model = model.cuda()
        dummy = dummy.cuda()
        with torch.no_grad():
            out = model(dummy)
        print(f"  ✅ GPU forward OK")
        print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

    print("\n" + "=" * 60)
    print("✅ Model test completed successfully!")
    print("=" * 60)

    return model


if __name__ == "__main__":
    test_model()