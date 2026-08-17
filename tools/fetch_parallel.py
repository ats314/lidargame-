"""Fetch one large file over several ranged connections at once.

The Vienna test archive is 4.2 GB and a single connection to wien.gv.at runs at
about 0.4 MB/s, which is three hours. The server sets `Accept-Ranges: bytes`, so
the fix is to ask for several slices at once and write them into one
pre-allocated file. Each worker owns a disjoint byte range and seeks to it, so
there is no coordination beyond the split.

Two things this has to get right, both learned the hard way:

**A short response is not a finished slice.** wien.gv.at closes the connection
at around 110 MB regardless of the range asked for. The first version of this
treated an empty read as completion, so all twelve workers stopped at 31% and
left a 4.2 GB file that was exactly the right length and full of holes -- which
reads as a corrupt download, not as an incomplete one. A worker now exits only
when its slice is actually full.

**Resumption has to measure, not assume.** Because the file is pre-allocated to
full length, its size says nothing about progress. What is already present is
found by scanning each slice for its filled prefix, so an interrupted run
resumes rather than restarting. At this size that is the difference between
minutes and hours.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.request

CHUNK = 1 << 20
#: Granularity of the resume scan. Small enough to waste little on a re-fetch,
#: large enough that scanning 4 GB is a few hundred reads.
PROBE = 1 << 16
MAX_STALLS = 8


def total_size(url: str) -> int:
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
    if "/" not in content_range:
        raise RuntimeError(f"server did not report a size: {content_range!r}")
    return int(content_range.rsplit("/", 1)[1])


def filled_prefix(path: str, start: int, end: int) -> int:
    """How much of [start, end] is already written.

    Binary search for the last non-zero probe block. A file legitimately
    containing a run of zeroes would be under-counted and simply re-fetched, so
    the failure mode is wasted bandwidth rather than a corrupt result.
    """
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as handle:
        handle.seek(start)
        if not any(handle.read(PROBE)):
            return 0
        low, high = start, end
        while low < high:
            mid = (low + high + 1) // 2
            handle.seek(mid)
            if any(handle.read(PROBE)):
                low = mid
            else:
                high = mid - 1
    return min(low + PROBE, end + 1) - start


def _worker(url: str, path: str, start: int, end: int, done: list, index: int,
            errors: list) -> None:
    """Fill [start, end] of `path`, resuming from whatever is already there."""
    need = end - start + 1
    stalls = 0
    while done[index] < need:
        before = done[index]
        try:
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={start + done[index]}-{end}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                with open(path, "r+b") as handle:
                    handle.seek(start + done[index])
                    while done[index] < need:
                        block = response.read(CHUNK)
                        if not block:
                            break          # short response -- reconnect, do not finish
                        handle.write(block)
                        done[index] += len(block)
        except Exception as exc:                       # noqa: BLE001
            if stalls >= MAX_STALLS:
                errors.append(f"slice {index}: {exc}")
                return
        if done[index] > before:
            stalls = 0                                 # progress resets patience
        else:
            stalls += 1
            if stalls > MAX_STALLS:
                errors.append(f"slice {index}: stalled at "
                              f"{done[index]:,}/{need:,} bytes")
                return
            time.sleep(min(2 ** stalls, 30))


def fetch(url: str, path: str, *, workers: int = 8, quiet: bool = False) -> int:
    size = total_size(url)
    exists = os.path.exists(path)
    with open(path, "r+b" if exists else "wb") as handle:
        handle.truncate(size)

    span = size // workers
    bounds = [(i * span, (i + 1) * span - 1 if i < workers - 1 else size - 1)
              for i in range(workers)]
    done = [filled_prefix(path, start, end) for start, end in bounds]
    if sum(done) and not quiet:
        print(f"resuming: {sum(done)/1e9:.2f}/{size/1e9:.2f} GB already present",
              flush=True)

    errors: list = []
    threads = [threading.Thread(target=_worker,
                                args=(url, path, start, end, done, i, errors),
                                daemon=True)
               for i, (start, end) in enumerate(bounds)]
    for thread in threads:
        thread.start()

    started, last = time.time(), sum(done)
    while any(thread.is_alive() for thread in threads):
        time.sleep(10)
        got = sum(done)
        if not quiet:
            print(f"  {got/1e9:.2f}/{size/1e9:.2f} GB  {100*got/size:5.1f}%  "
                  f"{(got-last)/10/1e6:.1f} MB/s  {time.time()-started:.0f}s",
                  flush=True)
        last = got
    for thread in threads:
        thread.join()

    if errors:
        raise RuntimeError("; ".join(errors))
    short = size - sum(done)
    if short:
        raise RuntimeError(f"{short:,} bytes missing after all workers finished")
    return size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("path")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    got = fetch(args.url, args.path, workers=args.workers)
    print(f"{args.path}: {got:,} bytes")
    sys.exit(0)
