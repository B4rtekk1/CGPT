# Kernel profile

- Date: 2026-08-20 08:54:11 Środkowoeuropejski czas letni
- GPU memory peak reference: 192 GB/s
- Raw log: `kernel_profile.log`

| Kernel | Average ms | Launches/s | GB/s | % of peak |
|---|---:|---:|---:|---:|
| RMSNorm kernel | 0.0578 | 17293.6 | 145.2 | 75.6% |
| RMSNorm BF16 kernel | 0.0540 | 18503.4 | 155.4 | 80.9% |
| RMSNorm backward kernel | 0.1425 | 7016.2 | 147.2 | 76.7% |
| RMSNorm BF16 backward kernel | 0.1350 | 7408.4 | 155.4 | 81.0% |
| Linear kernel | 1.0593 | 944.1 | 39.6 | 20.6% |
| SwiGLU kernel | 0.0733 | 13644.8 | 171.7 | 89.4% |
| RoPE kernel | 0.0643 | 15545.6 | 165.0 | 86.0% |
| RoPE backward kernel | 0.0665 | 15036.5 | 159.6 | 83.1% |
| RoPE BF16 kernel | 0.0657 | 15222.3 | 161.6 | 84.2% |
| RoPE BF16 backward kernel | 0.0638 | 15676.7 | 166.4 | 86.7% |
| Flash attention kernel | 0.4078 | 2452.0 | 25.7 | 13.4% |
| Flash attention LSE forward kernel | 0.4062 | 2461.6 | 26.0 | 13.5% |
| Flash attention backward kernel | 1.9758 | 506.1 | 8.5 | 4.4% |
| Flash attention LSE backward kernel | 1.6355 | 611.4 | 12.9 | 6.7% |
| Embedding kernel | 0.0502 | 19933.8 | 167.2 | 87.1% |
| Embedding backward kernel | 0.0766 | 13047.0 | 164.2 | 85.5% |
| Cross entropy kernel | 3.1380 | 318.7 | 65.6 | 34.2% |
