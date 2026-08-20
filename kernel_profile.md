# Kernel profile

- Date: 2026-08-20 08:41:04 Środkowoeuropejski czas letni
- GPU memory peak reference: 192 GB/s
- Raw log: `kernel_profile.log`

| Kernel | Average ms | Launches/s | GB/s | % of peak |
|---|---:|---:|---:|---:|
| RMSNorm kernel | 0.0493 | 20292.2 | 170.4 | 88.7% |
| RMSNorm BF16 kernel | 0.0500 | 20016.8 | 168.1 | 87.5% |
| RMSNorm backward kernel | 0.1226 | 8157.5 | 171.1 | 89.1% |
| RMSNorm BF16 backward kernel | 0.1238 | 8078.8 | 169.5 | 88.3% |
| Linear kernel | 1.0397 | 961.8 | 40.3 | 21.0% |
| SwiGLU kernel | 0.0732 | 13669.6 | 172.0 | 89.6% |
| RoPE kernel | 0.0638 | 15683.6 | 166.5 | 86.7% |
| RoPE backward kernel | 0.0640 | 15614.5 | 165.8 | 86.3% |
| RoPE BF16 kernel | 0.0656 | 15252.5 | 161.9 | 84.3% |
| RoPE BF16 backward kernel | 0.0636 | 15732.2 | 167.0 | 87.0% |
| Flash attention kernel | 0.4060 | 2463.3 | 25.8 | 13.5% |
| Flash attention LSE forward kernel | 0.4042 | 2474.3 | 26.1 | 13.6% |
| Flash attention backward kernel | 1.9718 | 507.2 | 8.5 | 4.4% |
| Flash attention LSE backward kernel | 1.6245 | 615.6 | 12.9 | 6.7% |
| Embedding kernel | 0.0501 | 19941.0 | 167.3 | 87.1% |
| Embedding backward kernel | 0.0765 | 13071.7 | 164.5 | 85.7% |
| Cross entropy kernel | 3.0553 | 327.3 | 67.4 | 35.1% |
