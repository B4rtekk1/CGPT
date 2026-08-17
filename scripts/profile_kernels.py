"""Run the CUDA kernel benchmarks and write a Markdown summary on any OS."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
from pathlib import Path


TESTS = (
    "rmsnorm_test", "linear_test", "linear_backward_test", "swiglu_test",
    "rope_test", "tensor_test", "device_buffer_test", "dtype_test",
    "flash_attention_test", "embedding_test", "cross_entropy_test",
    "cuda_graph_test", "cuda_benchmark_test", "kv_cache_test",
    "transformer_block_test", "transformer_model_test", "adamw_test",
)

TRAFFIC_MB = {
    "RMSNorm kernel": 8.3968, "RMSNorm BF16 kernel": 8.3968,
    "RMSNorm backward kernel": 20.98, "RMSNorm BF16 backward kernel": 20.98,
    "Linear kernel": 41.943, "SwiGLU kernel": 12.583,
    "RoPE kernel": 10.617, "RoPE backward kernel": 10.617,
    "RoPE BF16 kernel": 10.617, "RoPE BF16 backward kernel": 10.617,
    "Flash attention kernel": 10.486, "Flash attention LSE forward kernel": 10.551,
    "Flash attention backward kernel": 16.777,
    "Flash attention LSE backward kernel": 21.036,
    "Embedding kernel": 8.389, "Embedding backward kernel": 12.583,
    "Cross entropy kernel": 205.85,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("kernel_profile.md"))
    parser.add_argument("--peak-memory-gbps", type=float, default=192.0)
    args = parser.parse_args()

    suffix = ".exe" if __import__("os").name == "nt" else ""
    build_dir = args.build_dir.resolve()
    runner = build_dir / f"run_all_tests{suffix}"
    if not runner.is_file():
        parser.error(f"Profiler runner not found: {runner}")

    command = []
    for test in TESTS:
        command += [test, str(build_dir / f"{test}{suffix}")]

    result = subprocess.run([str(runner), *command], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
    output_path = args.output.resolve()
    log_path = output_path.with_suffix(".log")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")

    rows = []
    pattern = re.compile(r"^\s*(.+?kernel)\s+\|\s+([0-9]+(?:\.[0-9]+)?)\s*$")
    for match in map(pattern.match, output.splitlines()):
        if not match:
            continue
        name, milliseconds = match.group(1).strip(), float(match.group(2))
        if name.startswith("CUPTI:") or milliseconds <= 0:
            continue
        launches = 1000.0 / milliseconds
        traffic = TRAFFIC_MB.get(name)
        gbps = traffic / milliseconds if traffic else None
        percent = 100.0 * gbps / args.peak_memory_gbps if gbps else None
        rows.append((name, milliseconds, launches, gbps, percent))

    if not rows:
        raise RuntimeError(f"Profiler returned no benchmarks. Raw log: {log_path}")

    lines = ["# Kernel profile", "",
             f"- Date: {dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}",
             f"- GPU memory peak reference: {args.peak_memory_gbps:g} GB/s",
             f"- Raw log: `{log_path.name}`", "",
             "| Kernel | Average ms | Launches/s | GB/s | % of peak |",
             "|---|---:|---:|---:|---:|"]
    for name, ms, launches, gbps, percent in rows:
        lines.append(f"| {name} | {ms:.4f} | {launches:.1f} | "
                     f"{gbps:.1f} | {percent:.1f}% |" if gbps else
                     f"| {name} | {ms:.4f} | {launches:.1f} | - | - |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Markdown table: {output_path}")
    print(f"Raw log:        {log_path}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
