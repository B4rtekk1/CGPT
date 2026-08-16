# Kernel profile

- Date: 2026-08-16 11:14:24
- GPU memory peak reference: 192 GB/s
- Raw log: `kernel_profile.log`

| Kernel | Average ms | Launches/s | GB/s | % of 192 GB/s |
|---|---:|---:|---:|---:|
| RMSNorm kernel | 0,0495 | 20 196,3 | 169,6 | 88,3% |
| RMSNorm BF16 kernel | 0,0510 | 19 609,8 | 164,7 | 85,8% |
| RMSNorm backward kernel | 0,1348 | 7 418,8 | 93,5 | 48,7% |
| RMSNorm BF16 backward kernel | 0,1441 | 6 941,5 | 87,5 | 45,6% |
| Linear kernel | 1,0746 | 930,5 | 39,0 | 20,3% |
| SwiGLU kernel | 0,0731 | 13 688,9 | 172,2 | 89,7% |
| RoPE kernel | 0,0688 | 14 537,4 | 154,3 | 80,4% |
| RoPE backward kernel | 0,0648 | 15 436,6 | 163,9 | 85,4% |
| RoPE BF16 kernel | 0,0664 | 15 062,1 | 159,9 | 83,3% |
| RoPE BF16 backward kernel | 0,0646 | 15 487,8 | 164,4 | 85,6% |
| Flash attention kernel | 2,8086 | 356,0 | 3,7 | 1,9% |
| Flash attention LSE forward kernel | 2,8011 | 357,0 | 3,8 | 2,0% |
| Flash attention backward kernel | 55,0760 | 18,2 | 0,3 | 0,2% |
| Flash attention LSE backward kernel | 131,8454 | 7,6 | 0,2 | 0,1% |
| Embedding kernel | 0,0493 | 20 291,8 | 170,2 | 88,7% |
| Embedding backward kernel | 0,0762 | 13 122,8 | 165,1 | 86,0% |
| Cross entropy kernel | 3,1798 | 314,5 | 64,7 | 33,7% |
