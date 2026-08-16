# Kernel profile

- Date: 2026-08-16 12:30:24
- GPU memory peak reference: 192 GB/s
- Raw log: `kernel_profile.log`

| Kernel | Average ms | Launches/s | GB/s | % of 192 GB/s |
|---|---:|---:|---:|---:|
| RMSNorm kernel | 0,0496 | 20 169,0 | 169,4 | 88,2% |
| RMSNorm BF16 kernel | 0,0502 | 19 903,3 | 167,1 | 87,0% |
| RMSNorm backward kernel | 0,1246 | 8 026,8 | 168,4 | 87,7% |
| RMSNorm BF16 backward kernel | 0,1282 | 7 802,0 | 163,7 | 85,3% |
| Linear kernel | 1,0433 | 958,5 | 40,2 | 20,9% |
| SwiGLU kernel | 0,0731 | 13 675,2 | 172,1 | 89,6% |
| RoPE kernel | 0,0639 | 15 646,3 | 166,1 | 86,5% |
| RoPE backward kernel | 0,0639 | 15 657,3 | 166,2 | 86,6% |
| RoPE BF16 kernel | 0,0653 | 15 310,2 | 162,5 | 84,7% |
| RoPE BF16 backward kernel | 0,0638 | 15 664,7 | 166,3 | 86,6% |
| Flash attention kernel | 2,7709 | 360,9 | 3,8 | 2,0% |
| Flash attention LSE forward kernel | 2,8338 | 352,9 | 3,7 | 1,9% |
| Flash attention backward kernel | 54,9384 | 18,2 | 0,3 | 0,2% |
| Flash attention LSE backward kernel | 41,0328 | 24,4 | 0,5 | 0,3% |
| Embedding kernel | 0,0492 | 20 320,2 | 170,5 | 88,8% |
| Embedding backward kernel | 0,0750 | 13 339,9 | 167,9 | 87,4% |
| Cross entropy kernel | 3,0545 | 327,4 | 67,4 | 35,1% |
