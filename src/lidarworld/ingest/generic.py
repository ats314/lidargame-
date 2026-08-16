"""Adapters for the interchange formats every point-cloud tool can emit:
PCD (PCL), PLY (CloudCompare / MeshLab / Open3D) and plain ASCII XYZ/CSV.

These carry no semantics, so the inference stage has to earn them from geometry
-- which is the honest path for most data people actually have lying around.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

from ..types import PointCloud, Source
from .base import IngestResult, register, remap

_PCD_TYPES = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "<u1", ("U", 2): "<u2",
              ("U", 4): "<u4", ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4"}


def _decompress_lzf(src: bytes, expected: int) -> bytes:
    """LZF decompressor for PCD ``binary_compressed`` payloads."""
    out = bytearray(expected)
    i = o = 0
    n = len(src)
    while i < n:
        ctrl = src[i]; i += 1
        if ctrl < 32:                       # literal run
            length = ctrl + 1
            out[o:o + length] = src[i:i + length]
            i += length; o += length
        else:                               # back reference
            length = ctrl >> 5
            if length == 7:
                length += src[i]; i += 1
            ref = o - ((ctrl & 0x1F) << 8) - src[i] - 1
            i += 1
            if ref < 0:
                raise ValueError("corrupt LZF stream")
            for _ in range(length + 2):
                out[o] = out[ref]
                o += 1; ref += 1
    return bytes(out[:o])


@register("pcd", (".pcd",), "PCL point cloud (ascii, binary, binary_compressed)")
def load_pcd(path: Path, options: dict) -> IngestResult:
    raw = path.read_bytes()
    header_end = raw.find(b"DATA")
    if header_end < 0:
        raise ValueError(f"{path}: no DATA line, not a PCD file")
    line_end = raw.find(b"\n", header_end)
    header_text = raw[:line_end].decode("ascii", "replace")
    body = raw[line_end + 1:]

    def field(name, cast=str, default=None):
        m = re.search(rf"^{name}\s+(.*)$", header_text, re.M)
        if not m:
            return default
        parts = m.group(1).split()
        return [cast(p) for p in parts] if len(parts) > 1 else cast(parts[0])

    names = field("FIELDS") or ["x", "y", "z"]
    if isinstance(names, str): names = [names]
    sizes = field("SIZE", int) or [4] * len(names)
    if isinstance(sizes, int): sizes = [sizes]
    types = field("TYPE") or ["F"] * len(names)
    if isinstance(types, str): types = [types]
    counts = field("COUNT", int) or [1] * len(names)
    if isinstance(counts, int): counts = [counts]
    n_points = int(field("POINTS", int) or (field("WIDTH", int) or 0) * (field("HEIGHT", int) or 1))
    data_mode = field("DATA") or "ascii"

    if data_mode == "ascii":
        total = sum(counts)
        flat = np.asarray(body.split(), dtype=np.float64)
        table = flat[: (flat.size // total) * total].reshape(-1, total)
        cols = {}
        c = 0
        for name, cnt in zip(names, counts):
            cols[name] = table[:, c] if cnt == 1 else table[:, c:c + cnt]
            c += cnt
    else:
        if data_mode == "binary_compressed":
            comp_size, uncomp_size = struct.unpack_from("<II", body, 0)
            payload = _decompress_lzf(body[8:8 + comp_size], uncomp_size)
            # binary_compressed stores each field contiguously, not interleaved.
            cols = {}
            pos = 0
            for name, size, typ, cnt in zip(names, sizes, types, counts):
                dt = np.dtype(_PCD_TYPES[(typ, size)])
                width = n_points * cnt * size
                arr = np.frombuffer(payload, dtype=dt, count=n_points * cnt, offset=pos)
                cols[name] = arr if cnt == 1 else arr.reshape(n_points, cnt)
                pos += width
        else:
            dtype = np.dtype([(name, _PCD_TYPES[(typ, size)], (cnt,) if cnt > 1 else ())
                              for name, size, typ, cnt in zip(names, sizes, types, counts)])
            table = np.frombuffer(body, dtype=dtype, count=n_points)
            cols = {name: table[name] for name in names}

    xyz = np.column_stack([np.asarray(cols[a], dtype=np.float64) for a in ("x", "y", "z")])
    finite = np.isfinite(xyz).all(axis=1)
    cloud = PointCloud(xyz[finite])
    for key in ("intensity", "i"):
        if key in cols:
            v = np.asarray(cols[key], dtype=np.float32)[finite]
            hi = float(v.max(initial=1.0))
            cloud["intensity"] = (v / hi if hi > 1.5 else v).astype(np.float32)
            break
    cloud.meta["pcd_fields"] = names
    source = Source(id=options.get("source_id", path.stem), uri=str(path),
                    license=options.get("license", "unknown"),
                    sensor=options.get("sensor", ""), notes=f"PCD {data_mode}, fields {names}")
    return IngestResult(cloud, source)


@register("ply", (".ply",), "PLY point cloud (ascii and binary little-endian)")
def load_ply(path: Path, options: dict) -> IngestResult:
    raw = path.read_bytes()
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError(f"{path}: no end_header")
    header = raw[:end].decode("ascii", "replace").splitlines()
    body = raw[raw.find(b"\n", end) + 1:]
    fmt = "ascii"
    count = 0
    props: list[tuple[str, str]] = []
    in_vertex = False
    ply_types = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
                 "uchar": "<u1", "uint8": "<u1", "char": "<i1", "int8": "<i1",
                 "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
                 "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}
    for line in header:
        parts = line.split()
        if not parts: continue
        if parts[0] == "format": fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex: count = int(parts[2])
        elif parts[0] == "property" and in_vertex and parts[1] != "list":
            props.append((parts[2], ply_types.get(parts[1], "<f4")))

    if fmt == "ascii":
        table = np.array(body.split()[: count * len(props)], dtype=np.float64).reshape(count, len(props))
        cols = {name: table[:, i] for i, (name, _) in enumerate(props)}
    elif fmt == "binary_little_endian":
        dtype = np.dtype([(name, t) for name, t in props])
        table = np.frombuffer(body, dtype=dtype, count=count)
        cols = {name: table[name] for name, _ in props}
    else:
        raise ValueError(f"{path}: unsupported PLY format {fmt!r}")

    xyz = np.column_stack([np.asarray(cols[a], dtype=np.float64) for a in ("x", "y", "z")])
    cloud = PointCloud(xyz)
    if "intensity" in cols:
        v = np.asarray(cols["intensity"], dtype=np.float32)
        cloud["intensity"] = v / max(float(v.max(initial=1.0)), 1.0)
    elif "red" in cols:
        rgb = np.column_stack([cols[c] for c in ("red", "green", "blue")]).astype(np.float32)
        cloud["intensity"] = (rgb.mean(axis=1) / 255.0).astype(np.float32)

    label_note = "unlabelled -- semantics will be inferred from geometry"
    ids = _label_column(cols)
    if ids is not None:
        cloud["source_class"] = np.clip(ids, 0, 255).astype(np.uint8)
        vocab, note = _resolve_vocabulary(ids, options)
        label_note = note
        if vocab is not None:
            from ..semantics.vocab import VOCABULARIES
            cloud["semantic"] = remap(ids, VOCABULARIES[vocab])

    source = Source(id=options.get("source_id", path.stem), uri=str(path),
                    license=options.get("license", "unknown"),
                    notes=f"PLY {fmt}, {len(props)} properties; {label_note}")
    return IngestResult(cloud, source)


#: Names the annotated PLY benchmarks use for their per-point class column.
_LABEL_NAMES = ("label", "class", "classification", "scalar_label", "scalar_Label",
                "scalar_class", "semantic", "sem_class", "category")


def _label_column(cols: dict) -> np.ndarray | None:
    for name in _LABEL_NAMES:
        if name in cols:
            return np.rint(np.asarray(cols[name], dtype=np.float64)).astype(np.int64)
    return None


def _resolve_vocabulary(ids: np.ndarray, options: dict) -> tuple[str | None, str]:
    """Pick the class vocabulary, preferring what the caller said over a guess.

    Every benchmark numbered its classes independently and several of the ranges
    collide outright -- DALES and Toronto-3D are both 0-8 and mean different
    things by 3. Guessing wrong silently relabels a whole dataset, so an
    ambiguous file is left unlabelled with a message naming the flag to pass.
    """
    from ..semantics.vocab import VOCABULARIES, coverage, detect

    asked = options.get("vocab") or options.get("vocabulary")
    if asked:
        if asked not in VOCABULARIES:
            raise ValueError(f"unknown label vocabulary {asked!r}; "
                             f"have {sorted(VOCABULARIES)}")
        got = coverage(asked, np.unique(ids))
        return asked, (f"{asked} labels ({got:.0%} of ids mapped)" if got > 0.5 else
                       f"{asked} labels, but only {got:.0%} of ids are in that "
                       "vocabulary -- check it is the right one")
    guess, score = detect(ids)
    if guess is None:
        return None, ("carries a label column this reader cannot attribute to a "
                      "dataset; pass --vocab (one of "
                      f"{', '.join(sorted(VOCABULARIES))}) to use it")
    return guess, f"{guess} labels, detected ({score:.0%} of ids mapped)"


@register("xyz", (".xyz", ".txt", ".csv", ".asc", ".pts"), "ASCII columns: x y z [intensity] [class]")
def load_xyz(path: Path, options: dict) -> IngestResult:
    delim = options.get("delimiter")
    skip = options.get("skip_rows", 0)
    with open(path, "r") as fh:
        first = fh.readline()
    if delim is None:
        delim = "," if first.count(",") >= 2 else (";" if first.count(";") >= 2 else None)
    if re.search(r"[a-df-zA-DF-Z]", first.replace("e", "").replace("E", "")):
        skip = max(skip, 1)                       # header row
    table = np.loadtxt(path, delimiter=delim, skiprows=skip, ndmin=2)
    if table.shape[1] < 3:
        raise ValueError(f"{path}: need at least 3 columns, found {table.shape[1]}")
    cloud = PointCloud(table[:, :3])
    if table.shape[1] >= 4:
        v = table[:, 3].astype(np.float32)
        hi = float(np.abs(v).max(initial=1.0))
        cloud["intensity"] = (v / hi if hi > 1.5 else v).astype(np.float32)
    if table.shape[1] >= 5:
        from .las import ASPRS
        from .base import remap
        cloud["semantic"] = remap(table[:, 4].astype(np.int32), ASPRS)
    source = Source(id=options.get("source_id", path.stem), uri=str(path),
                    license=options.get("license", "unknown"),
                    notes=f"ASCII, {table.shape[1]} columns")
    return IngestResult(cloud, source)
