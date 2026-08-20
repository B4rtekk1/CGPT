"""Download a size-limited streaming sample from FineWeb-Edu."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fineweb_edu_100mb.jsonl"),
        help="Output file (default: data/fineweb_edu_100mb.jsonl)",
    )
    parser.add_argument(
        "--size-mb",
        type=float,
        default=100,
        help="Target size in MiB (default: 100)",
    )
    parser.add_argument(
        "--config",
        default="sample-10BT",
        help="FineWeb-Edu configuration (default: sample-10BT)",
    )
    args = parser.parse_args()

    if args.size_mb <= 0:
        parser.error("--size-mb must be greater than zero")

    target_bytes = int(args.size_mb * 1024 * 1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=args.config,
            split="train",
            streaming=True,
        )
    except ConnectionError as error:
        raise SystemExit(
            "Cannot connect to Hugging Face Hub. Check your internet connection, "
            "proxy/VPN, and whether HF_HUB_OFFLINE=1 is not set. "
            "Update libraries with: "
            "python -m pip install -U datasets huggingface_hub fsspec"
        ) from error

    written = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for example in dataset:
            line = json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n"
            encoded = line.encode("utf-8")

            if written + len(encoded) > target_bytes:
                break

            output.write(line)
            written += len(encoded)

    print(f"Written {written / (1024 * 1024):.2f} MiB to: {args.output}")


if __name__ == "__main__":
    main()
