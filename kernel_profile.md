# Kernel profile

- Date: 2026-08-17 12:36:23
- GPU memory peak reference: 192 GB/s
- Raw log: `kernel_profile.log`

| Kernel | Average ms | Launches/s | GB/s | % of 192 GB/s |
|---|---:|---:|---:|---:|
| RMSNorm kernel | 0,0492 | 20 323,6 | 170,7 | 88,9% |
| RMSNorm BF16 kernel | 0,0497 | 20 110,2 | 168,9 | 87,9% |
| RMSNorm backward kernel | 0,1226 | 8 153,7 | 171,1 | 89,1% |
| RMSNorm BF16 backward kernel | 0,1229 | 8 135,8 | 170,7 | 88,9% |
| Linear kernel | 1,0365 | 964,8 | 40,5 | 21,1% |
| SwiGLU kernel | 0,0730 | 13 700,1 | 172,4 | 89,8% |
| RoPE kernel | 0,0644 | 15 523,1 | 164,8 | 85,8% |
| RoPE backward kernel | 0,0641 | 15 601,8 | 165,6 | 86,3% |
| RoPE BF16 kernel | 0,0655 | 15 268,6 | 162,1 | 84,4% |
| RoPE BF16 backward kernel | 0,0639 | 15 639,4 | 166,0 | 86,5% |
| Flash attention kernel | 2,7624 | 362,0 | 3,8 | 2,0% |
| Flash attention LSE forward kernel | 2,7978 | 357,4 | 3,8 | 2,0% |
| Flash attention backward kernel | 54,8921 | 18,2 | 0,3 | 0,2% |
| Flash attention LSE backward kernel | 40,9947 | 24,4 | 0,5 | 0,3% |
| Embedding kernel | 0,0497 | 20 137,7 | 168,9 | 88,0% |
| Embedding backward kernel | 0,0751 | 13 318,4 | 167,6 | 87,3% |
| Cross entropy kernel | 3,0538 | 327,5 | 67,4 | 35,1% |
