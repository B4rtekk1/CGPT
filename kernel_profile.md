# Kernel profile

- Date: 2026-08-18 08:12:37 Środkowoeuropejski czas letni
- GPU memory peak reference: 192 GB/s
- Raw log: `kernel_profile.log`

| Kernel | Average ms | Launches/s | GB/s | % of peak |
|---|---:|---:|---:|---:|
| RMSNorm kernel | 0.0493 | 20296.7 | 170.4 | 88.8% |
| RMSNorm BF16 kernel | 0.0508 | 19694.3 | 165.4 | 86.1% |
| RMSNorm backward kernel | 0.1266 | 7897.5 | 165.7 | 86.3% |
| RMSNorm BF16 backward kernel | 0.1347 | 7421.3 | 155.7 | 81.1% |
| Linear kernel | 1.0333 | 967.8 | 40.6 | 21.1% |
| SwiGLU kernel | 0.0731 | 13670.7 | 172.0 | 89.6% |
| RoPE kernel | 0.0642 | 15572.2 | 165.3 | 86.1% |
| RoPE backward kernel | 0.0640 | 15620.9 | 165.8 | 86.4% |
| RoPE BF16 kernel | 0.0654 | 15279.5 | 162.2 | 84.5% |
| RoPE BF16 backward kernel | 0.0638 | 15665.1 | 166.3 | 86.6% |
| Flash attention kernel | 2.7577 | 362.6 | 3.8 | 2.0% |
| Flash attention LSE forward kernel | 2.7945 | 357.9 | 3.8 | 2.0% |
| Flash attention backward kernel | 54.8536 | 18.2 | 0.3 | 0.2% |
| Flash attention LSE backward kernel | 41.0080 | 24.4 | 0.5 | 0.3% |
| Embedding kernel | 0.0501 | 19951.3 | 167.4 | 87.2% |
| Embedding backward kernel | 0.0769 | 13002.2 | 163.6 | 85.2% |
| Cross entropy kernel | 3.0566 | 327.2 | 67.3 | 35.1% |
