# Kernel profile

- Date: 2026-08-19 20:02:13 Środkowoeuropejski czas letni
- GPU memory peak reference: 192 GB/s
- Raw log: `kernel_profile.log`

| Kernel | Average ms | Launches/s | GB/s | % of peak |
|---|---:|---:|---:|---:|
| RMSNorm kernel | 0.0497 | 20129.2 | 169.0 | 88.0% |
| RMSNorm BF16 kernel | 0.0504 | 19831.4 | 166.5 | 86.7% |
| RMSNorm backward kernel | 0.1231 | 8124.7 | 170.5 | 88.8% |
| RMSNorm BF16 backward kernel | 0.1239 | 8071.4 | 169.3 | 88.2% |
| Linear kernel | 1.0375 | 963.9 | 40.4 | 21.1% |
| SwiGLU kernel | 0.0730 | 13698.6 | 172.4 | 89.8% |
| RoPE kernel | 0.0668 | 14973.2 | 159.0 | 82.8% |
| RoPE backward kernel | 0.0675 | 14821.4 | 157.4 | 82.0% |
| RoPE BF16 kernel | 0.0753 | 13288.7 | 141.1 | 73.5% |
| RoPE BF16 backward kernel | 0.0748 | 13373.3 | 142.0 | 74.0% |
| Flash attention kernel | 2.8046 | 356.6 | 3.7 | 1.9% |
| Flash attention LSE forward kernel | 2.8052 | 356.5 | 3.8 | 2.0% |
| Flash attention backward kernel | 54.9566 | 18.2 | 0.3 | 0.2% |
| Flash attention LSE backward kernel | 40.9636 | 24.4 | 0.5 | 0.3% |
| Embedding kernel | 0.0497 | 20107.4 | 168.7 | 87.9% |
| Embedding backward kernel | 0.0770 | 12979.3 | 163.3 | 85.1% |
| Cross entropy kernel | 3.0549 | 327.3 | 67.4 | 35.1% |
