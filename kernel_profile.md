# Kernel profile

- Date: 2026-08-16 12:08:29
- GPU memory peak reference: 192 GB/s
- Raw log: `kernel_profile.log`

| Kernel | Average ms | Launches/s | GB/s | % of 192 GB/s |
|---|---:|---:|---:|---:|
| RMSNorm kernel | 0,0501 | 19 944,2 | 167,5 | 87,2% |
| RMSNorm BF16 kernel | 0,0535 | 18 700,3 | 157,0 | 81,8% |
| RMSNorm backward kernel | 0,1248 | 8 013,6 | 101,0 | 52,6% |
| RMSNorm BF16 backward kernel | 0,1254 | 7 973,7 | 100,5 | 52,3% |
| Linear kernel | 1,0567 | 946,3 | 39,7 | 20,7% |
| SwiGLU kernel | 0,0731 | 13 675,2 | 172,1 | 89,6% |
| RoPE kernel | 0,0638 | 15 671,8 | 166,4 | 86,7% |
| RoPE backward kernel | 0,0641 | 15 595,8 | 165,6 | 86,2% |
| RoPE BF16 kernel | 0,0654 | 15 291,7 | 162,4 | 84,6% |
| RoPE BF16 backward kernel | 0,0636 | 15 729,0 | 167,0 | 87,0% |
| Flash attention kernel | 2,7801 | 359,7 | 3,8 | 2,0% |
| Flash attention LSE forward kernel | 2,8119 | 355,6 | 3,8 | 2,0% |
| Flash attention backward kernel | 55,0967 | 18,1 | 0,3 | 0,2% |
| Flash attention LSE backward kernel | 41,2009 | 24,3 | 0,5 | 0,3% |
| Embedding kernel | 0,0492 | 20 323,6 | 170,5 | 88,8% |
| Embedding backward kernel | 0,0750 | 13 339,6 | 167,9 | 87,4% |
| Cross entropy kernel | 3,1807 | 314,4 | 64,7 | 33,7% |
