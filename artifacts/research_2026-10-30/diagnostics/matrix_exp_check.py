"""Accuracy of torch.matrix_exp for one matrix against a batch (scratch check).

Reference: Taylor series in float64 summed to 40 terms (exact to rounding for the
1-norms used here, all below 1).
"""

import sys

import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
torch.manual_seed(0)


def reference(matrix: torch.Tensor) -> torch.Tensor:
    a = matrix.to(dtype=torch.float64, device="cpu")
    term = torch.eye(3, dtype=torch.float64)
    total = term.clone()
    for order in range(1, 41):
        term = term @ a / order
        total = total + term
    return total


devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
base = torch.randn(3, 3, dtype=torch.float64)
base = base + base.T
base = base / torch.linalg.matrix_norm(base, ord=1)
for device in devices:
    for dtype in (torch.float64, torch.float32):
        print(f"-- {device} {dtype}")
        for scale in (1e-4, 3e-4, 1e-3, 1e-2, 4e-2, 4.9e-2, 6e-2, 0.2, 0.5, 0.7):
            a = (base * scale).to(dtype=dtype, device=device)
            exact = reference(base * scale if dtype == torch.float64 else a.cpu())
            norm = float(torch.linalg.matrix_norm(exact, ord=1))
            gaps = []
            for label, value in (
                ("single", torch.matrix_exp(a)),
                ("batch1", torch.matrix_exp(a[None])[0]),
                ("batch2", torch.matrix_exp(torch.stack([a, a]))[0]),
                ("batch3", torch.matrix_exp(torch.stack([a, a * 0.5, a]))[0]),
            ):
                gap = float(
                    torch.linalg.matrix_norm(value.to(dtype=torch.float64, device="cpu") - exact, ord=1)
                )
                gaps.append(f"{label} {gap / norm:.3e}")
            print(f"norm {scale:.1e}: " + "  ".join(gaps))
sys.exit(0)
