# Kernel profile

- Date: 2026-08-16 19:21:35
- GPU memory peak reference: 192 GB/s
- Raw log: `kernel_profile.log`

| Kernel | Average ms | Launches/s | GB/s | % of 192 GB/s |
|---|---:|---:|---:|---:|
| RMSNorm kernel | 0,0496 | 20 167,4 | 169,3 | 88,2% |
| RMSNorm BF16 kernel | 0,0501 | 19 956,9 | 167,6 | 87,3% |
| RMSNorm backward kernel | 0,1228 | 8 145,4 | 170,9 | 89,0% |
| RMSNorm BF16 backward kernel | 0,1243 | 8 046,2 | 168,8 | 87,9% |
| Linear kernel | 1,0344 | 966,8 | 40,5 | 21,1% |
| SwiGLU kernel | 0,0731 | 13 683,8 | 172,2 | 89,7% |
| RoPE kernel | 0,0639 | 15 643,8 | 166,1 | 86,5% |
| RoPE backward kernel | 0,0640 | 15 627,9 | 165,9 | 86,4% |
| RoPE BF16 kernel | 0,0653 | 15 319,1 | 162,6 | 84,7% |
| RoPE BF16 backward kernel | 0,0638 | 15 666,4 | 166,3 | 86,6% |
| Flash attention kernel | 2,7606 | 362,2 | 3,8 | 2,0% |
| Flash attention LSE forward kernel | 2,7958 | 357,7 | 3,8 | 2,0% |
| Flash attention backward kernel | 54,8810 | 18,2 | 0,3 | 0,2% |
| Flash attention LSE backward kernel | 40,9945 | 24,4 | 0,5 | 0,3% |
| Embedding kernel | 0,0493 | 20 297,6 | 170,3 | 88,7% |
| Embedding backward kernel | 0,0751 | 13 309,4 | 167,5 | 87,2% |
| Cross entropy kernel | 3,0588 | 326,9 | 67,3 | 35,1% |
