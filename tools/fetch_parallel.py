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
import json
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


def _progress_path(path: str) -> str:
    return path + ".progress"


def load_progress(path: str, size: int, bounds: list) -> list | None:
    """Per-slice byte counts from the sidecar, or None if it cannot be trusted.

    Progress is recorded, not inferred. The previous version inferred it from
    file content and silently corrupted a 4.2 GB download: it binary-searched
    for the last non-zero 64 KiB probe block and then claimed `block + 64 KiB`
    bytes were present. When the written data ended partway into that block --
    which is the normal case, since a connection drops mid-buffer -- resume
    began up to 64 KiB too far along and left a hole. The file came out the
    right length, the outer zip parsed, and six of seven members failed CRC.
    The docstring at the time claimed the failure mode was "wasted bandwidth
    rather than a corrupt result", which was exactly backwards.

    A sidecar cannot drift the same way: it only ever records bytes already
    written and flushed.
    """
    sidecar = _progress_path(path)
    if not (os.path.exists(path) and os.path.exists(sidecar)):
        return None
    try:
        with open(sidecar) as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        return None
    if state.get("size") != size or state.get("bounds") != bounds:
        return None                       # different file or different split
    done = state.get("done")
    if not isinstance(done, list) or len(done) != len(bounds):
        return None
    if os.path.getsize(path) != size:
        return None
    return [int(v) for v in done]


def save_progress(path: str, size: int, bounds: list, done: list) -> None:
    """Write the sidecar atomically, so a kill mid-write cannot poison a resume."""
    sidecar = _progress_path(path)
    temporary = sidecar + ".tmp"
    with open(temporary, "w") as handle:
        json.dump({"size": size, "bounds": bounds, "done": list(done)}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, sidecar)


def verify_zip(path: str) -> list:
    """Members whose CRC does not match. Empty means the download is sound.

    Worth the read: a ranged parallel download can produce a file of exactly the
    right length that is wrong in the middle, and nothing about its size, its
    central directory or its member listing will say so.
    """
    import zipfile
    import zlib
    try:
        # Hamburg's archives are Deflate64. Without this every member read
        # raises NotImplementedError and the whole archive is reported corrupt
        # in about two seconds -- which is how a false alarm looks: far too fast
        # to have decompressed anything.
        import zipfile_deflate64                      # noqa: F401
    except ImportError:
        pass

    bad = []
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        return [f"<archive unreadable: {exc}>"]
    for info in archive.infolist():
        if info.is_dir():
            continue
        try:
            crc = 0
            with archive.open(info) as handle:
                while True:
                    block = handle.read(1 << 22)
                    if not block:
                        break
                    crc = zlib.crc32(block, crc)
            if (crc & 0xFFFFFFFF) != info.CRC:
                bad.append(info.filename)
        except Exception:                              # noqa: BLE001
            bad.append(info.filename)
    return bad


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
    bounds = [[i * span, (i + 1) * span - 1 if i < workers - 1 else size - 1]
              for i in range(workers)]
    done = load_progress(path, size, bounds)
    if done is None:
        # No trustworthy record of what is already there. Start over rather than
        # guess: a wrong guess produces a file of the right length with a hole
        # in it, and that is far more expensive than re-fetching.
        done = [0] * len(bounds)
        if not quiet and exists:
            print("no usable progress record -- restarting the download",
                  flush=True)
    elif sum(done) and not quiet:
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
        save_progress(path, size, bounds, done)
        if not quiet:
            print(f"  {got/1e9:.2f}/{size/1e9:.2f} GB  {100*got/size:5.1f}%  "
                  f"{(got-last)/10/1e6:.1f} MB/s  {time.time()-started:.0f}s",
                  flush=True)
        last = got
    for thread in threads:
        thread.join()
    save_progress(path, size, bounds, done)

    if errors:
        raise RuntimeError("; ".join(errors))
    short = size - sum(done)
    if short:
        raise RuntimeError(f"{short:,} bytes missing after all workers finished")
    if path.lower().endswith(".zip"):
        bad = verify_zip(path)
        if bad:
            raise RuntimeError(
                f"{len(bad)} zip members fail CRC after download, e.g. {bad[:3]}. "
                "The file is the right length and wrong in the middle.")
        if not quiet:
            print("  zip CRC verified", flush=True)
    os.remove(_progress_path(path))
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
