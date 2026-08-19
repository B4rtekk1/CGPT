# Kernel profile

- Date: 2026-08-19 20:36:38 Środkowoeuropejski czas letni
- GPU memory peak reference: 192 GB/s
- Raw log: `kernel_profile.log`

| Kernel | Average ms | Launches/s | GB/s | % of peak |
|---|---:|---:|---:|---:|
| RMSNorm kernel | 0.0499 | 20057.4 | 168.4 | 87.7% |
| RMSNorm BF16 kernel | 0.0507 | 19741.0 | 165.8 | 86.3% |
| RMSNorm backward kernel | 0.1335 | 7493.1 | 157.2 | 81.9% |
| RMSNorm BF16 backward kernel | 0.1312 | 7622.2 | 159.9 | 83.3% |
| Linear kernel | 1.0605 | 943.0 | 39.6 | 20.6% |
| SwiGLU kernel | 0.0732 | 13664.9 | 171.9 | 89.6% |
| RoPE kernel | 0.0639 | 15641.9 | 166.1 | 86.5% |
| RoPE backward kernel | 0.0647 | 15461.7 | 164.2 | 85.5% |
| RoPE BF16 kernel | 0.0665 | 15047.3 | 159.8 | 83.2% |
| RoPE BF16 backward kernel | 0.0639 | 15638.0 | 166.0 | 86.5% |
| Flash attention kernel | 2.7687 | 361.2 | 3.8 | 2.0% |
| Flash attention LSE forward kernel | 2.8746 | 347.9 | 3.7 | 1.9% |
| Flash attention backward kernel | 54.9813 | 18.2 | 0.3 | 0.2% |
| Flash attention LSE backward kernel | 41.0714 | 24.3 | 0.5 | 0.3% |
| Embedding kernel | 0.0502 | 19927.9 | 167.2 | 87.1% |
| Embedding backward kernel | 0.0768 | 13015.7 | 163.8 | 85.3% |
| Cross entropy kernel | 3.0571 | 327.1 | 67.3 | 35.1% |
