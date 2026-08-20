"""Download and decompress a fixed-size Wikimedia text sample."""

from __future__ import annotations

import argparse
import bz2
import sys
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_URL = (
    "https://dumps.wikimedia.org/enwiki/latest/"
    "enwiki-latest-pages-articles-multistream.xml.bz2"
)


def download_text(url: str, output: Path, size: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    decompressor = bz2.BZ2Decompressor()
    written = 0

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CGPT-tokenizer-benchmark/1.0"},
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        with output.open("wb") as destination:
            for compressed in iter(lambda: response.read(1024 * 1024), b""):
                pending = compressed
                while pending and written < size:
                    decoded = decompressor.decompress(pending)
                    pending = decompressor.unused_data

                    if decoded:
                        chunk = decoded[: size - written]
                        destination.write(chunk)
                        written += len(chunk)
                        print(
                            f"\rDownloaded: {written / 1024 / 1024:7.2f} / "
                            f"{size / 1024 / 1024:.2f} MiB",
                            end="",
                            flush=True,
                        )

                    if decompressor.eof:
                        if not pending:
                            break
                        decompressor = bz2.BZ2Decompressor()
                    elif not decoded and not pending:
                        break

                if written >= size:
                    break

    print()
    if written < size:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"The source ended after {written} bytes; {size} bytes were required."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("data/tokenizer_100mb.txt")
    )
    parser.add_argument(
        "--size-mib", type=int, default=100,
        help="Output size in MiB (default: 100).",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="URL of the .bz2 dump.")
    args = parser.parse_args()

    if args.size_mib <= 0:
        parser.error("--size-mib must be greater than zero")

    try:
        download_text(args.url, args.output, args.size_mib * 1024 * 1024)
    except (OSError, RuntimeError, urllib.error.URLError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1

    print(f"Saved {args.output} ({args.output.stat().st_size} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
