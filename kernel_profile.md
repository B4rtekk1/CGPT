# Kernel profile

- Date: 2026-08-20 09:17:40 Środkowoeuropejski czas letni
- GPU memory peak reference: 192 GB/s
- Raw log: `kernel_profile.log`

| Kernel | Average ms | Launches/s | GB/s | % of peak |
|---|---:|---:|---:|---:|
| RMSNorm kernel | 0.0492 | 20306.2 | 170.5 | 88.8% |
| RMSNorm BF16 kernel | 0.0498 | 20086.8 | 168.7 | 87.8% |
| RMSNorm backward kernel | 0.1228 | 8143.5 | 170.8 | 89.0% |
| RMSNorm BF16 backward kernel | 0.1230 | 8130.7 | 170.6 | 88.8% |
| Linear kernel | 1.0327 | 968.3 | 40.6 | 21.2% |
| SwiGLU kernel | 0.0729 | 13710.3 | 172.5 | 89.9% |
| RoPE kernel | 0.0641 | 15592.8 | 165.5 | 86.2% |
| RoPE backward kernel | 0.0640 | 15635.7 | 166.0 | 86.5% |
| RoPE BF16 kernel | 0.0653 | 15306.9 | 162.5 | 84.6% |
| RoPE BF16 backward kernel | 0.0639 | 15642.4 | 166.1 | 86.5% |
| Flash attention kernel | 0.4068 | 2458.5 | 25.8 | 13.4% |
| Flash attention LSE forward kernel | 0.4045 | 2472.2 | 26.1 | 13.6% |
| Flash attention backward kernel | 1.9464 | 513.8 | 8.6 | 4.5% |
| Flash attention LSE backward kernel | 1.6369 | 610.9 | 12.9 | 6.7% |
| Embedding kernel | 0.0502 | 19926.3 | 167.2 | 87.1% |
| Embedding backward kernel | 0.0766 | 13052.4 | 164.2 | 85.5% |
| Cross entropy kernel | 2.3196 | 431.1 | 88.7 | 46.2% |
