"""Read Wavefront OBJ + MTL, for photogrammetric city meshes.

Helsinki publishes its reality mesh as OBJ: 2 km tiles, each holding 250 m
subtiles, each subtile 64 OBJ chunks with a 1024x1024 JPEG apiece. That is a
different shape of data from CityGML -- no semantics, no surface classes, no
building ids, just triangles and texture -- and it is the shape that carries the
thing CityGML LoD3 cannot: a facade with actual depth. A balcony is geometry
here, not paint.

The reader is deliberately narrow. It handles what these files contain and
refuses what they do not, rather than being a general OBJ importer that
half-works on everything:

    v / vt / f          positions, texture coordinates, faces
    usemtl              material runs, which become texture groups
    mtllib -> map_Kd    the image per material

No normals (they are regenerated), no vertex colours, no smoothing groups, no
negative indices beyond the relative form OBJ allows, no quads assumed -- faces
are fanned, which is correct because these are already triangulated.

Coordinates are LOCAL. A Helsinki chunk's vertices read like (7687, 5043, 24),
which is neither ETRS-GK25 nor a tile-corner offset, so georeferencing has to
come from outside the file and is the caller's problem. Returning local
coordinates and saying so is better than guessing an origin: a mesh placed on
the wrong offset is a building in the wrong street and nothing about it raises.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Group:
    """A run of triangles sharing one material, and therefore one texture."""
    material: str
    image: Path | None
    faces: np.ndarray                      # (T, 3) indices into positions/uvs

    def __len__(self) -> int:
        return len(self.faces)


@dataclass
class Mesh:
    positions: np.ndarray                  # (N, 3) local coordinates
    uvs: np.ndarray                        # (N, 2), empty if untextured
    groups: list[Group] = field(default_factory=list)
    source: Path | None = None

    @property
    def triangles(self) -> int:
        return sum(len(g) for g in self.groups)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.positions.min(axis=0), self.positions.max(axis=0)

    @property
    def textured(self) -> int:
        return sum(len(g) for g in self.groups if g.image is not None)


def read_mtl(path: str | Path) -> dict[str, Path | None]:
    """material name -> diffuse texture path, resolved next to the .mtl."""
    path = Path(path)
    out: dict[str, Path | None] = {}
    if not path.exists():
        return out
    current = None
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "newmtl":
            current = parts[1]
            out.setdefault(current, None)
        elif parts[0].lower() == "map_kd" and current:
            # map_Kd can carry options before the filename; the name is last.
            candidate = path.parent / parts[-1]
            out[current] = candidate if candidate.exists() else None
    return out


def read_obj(path: str | Path) -> Mesh:
    """One OBJ chunk. UVs are indexed per-vertex by duplicating shared corners.

    OBJ indexes position and texture coordinate separately, so a corner shared
    between two triangles with different UVs is one v and two vt. glTF has a
    single index buffer, so those corners have to be split. Doing it here rather
    than in the exporter keeps the split where the information about it is.
    """
    path = Path(path)
    positions: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    materials: dict[str, Path | None] = {}
    runs: list[tuple[str, list[tuple[int, int]]]] = []
    current = ""

    with open(path, "r", errors="replace") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            tag, _, rest = line.partition(" ")
            if tag == "v":
                x, y, z = rest.split()[:3]
                positions.append((float(x), float(y), float(z)))
            elif tag == "vt":
                parts = rest.split()
                texcoords.append((float(parts[0]),
                                  float(parts[1]) if len(parts) > 1 else 0.0))
            elif tag == "usemtl":
                current = rest.strip()
                runs.append((current, []))
            elif tag == "mtllib":
                materials.update(read_mtl(path.parent / rest.strip()))
            elif tag == "f":
                corners = []
                for token in rest.split():
                    bits = token.split("/")
                    vi = int(bits[0])
                    ti = int(bits[1]) if len(bits) > 1 and bits[1] else 0
                    # OBJ is 1-based, and negative indices count back from the
                    # end of what has been read so far.
                    vi = vi - 1 if vi > 0 else len(positions) + vi
                    if ti:
                        ti = ti - 1 if ti > 0 else len(texcoords) + ti
                    else:
                        ti = -1
                    corners.append((vi, ti))
                if not runs:
                    runs.append((current, []))
                fan = runs[-1][1]
                for i in range(1, len(corners) - 1):
                    fan.append(corners[0])
                    fan.append(corners[i])
                    fan.append(corners[i + 1])

    position_array = np.asarray(positions, dtype=np.float64)
    texcoord_array = (np.asarray(texcoords, dtype=np.float32)
                      if texcoords else np.zeros((0, 2), dtype=np.float32))

    # Split shared corners that disagree about their UV.
    index_of: dict[tuple[int, int], int] = {}
    out_positions: list[np.ndarray] = []
    out_uvs: list[np.ndarray] = []
    groups: list[Group] = []

    for material, corners in runs:
        if not corners:
            continue
        faces = np.empty(len(corners), dtype=np.int64)
        for slot, key in enumerate(corners):
            found = index_of.get(key)
            if found is None:
                found = len(out_positions)
                index_of[key] = found
                out_positions.append(position_array[key[0]])
                out_uvs.append(texcoord_array[key[1]] if key[1] >= 0
                               else np.zeros(2, dtype=np.float32))
            faces[slot] = found
        groups.append(Group(material=material,
                            image=materials.get(material),
                            faces=faces.reshape(-1, 3)))

    return Mesh(
        positions=(np.asarray(out_positions, dtype=np.float64)
                   if out_positions else np.zeros((0, 3))),
        uvs=(np.asarray(out_uvs, dtype=np.float32)
             if out_uvs else np.zeros((0, 2), dtype=np.float32)),
        groups=groups, source=path)


def read_directory(directory: str | Path, *, limit: int | None = None) -> list[Mesh]:
    """Every OBJ chunk in a subtile directory, in name order."""
    directory = Path(directory)
    files = sorted(directory.glob("*.obj"))
    if limit is not None:
        files = files[:limit]
    return [read_obj(f) for f in files]


def merge(meshes: list[Mesh]) -> Mesh:
    """One mesh from many chunks, keeping material groups distinct.

    Chunks are merged rather than exported separately because a 250 m subtile is
    64 files and an engine should not see 64 draw calls where it could see the
    number of distinct textures.
    """
    meshes = [m for m in meshes if len(m.positions)]
    if not meshes:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 2)), [])
    positions, uvs, groups = [], [], []
    offset = 0
    for mesh in meshes:
        positions.append(mesh.positions)
        uvs.append(mesh.uvs if len(mesh.uvs) == len(mesh.positions)
                   else np.zeros((len(mesh.positions), 2), dtype=np.float32))
        for group in mesh.groups:
            groups.append(Group(group.material, group.image,
                                group.faces + offset))
        offset += len(mesh.positions)
    return Mesh(np.vstack(positions), np.vstack(uvs), groups)


#: A triangle bigger than this in a 250 m photogrammetric subtile is not a
#: surface. It is webbing -- a bridge the reconstruction threw across a street,
#: a courtyard or the sky because it could not tell that the gap was real.
#:
#: They are textured like everything else, which is why they are not obvious in
#: the data: the material has an image, and the triangle simply stretches a few
#: texels across hundreds of square metres. In a render that reads as a flat
#: grey membrane hanging over the street. Measured on the historic core, 60% of
#: all surface area sits in triangles over 5 m2.
WEBBING_MAX_AREA_M2 = 5.0


def triangle_areas(mesh: Mesh, faces: np.ndarray) -> np.ndarray:
    a, b, c = (mesh.positions[faces[:, i]] for i in range(3))
    return np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2.0


def drop_webbing(mesh: Mesh, max_area: float = WEBBING_MAX_AREA_M2) -> tuple[Mesh, dict]:
    """Remove reconstruction bridges. Returns the mesh and what was removed.

    Deleting them leaves holes, and the holes are correct: the gap it bridged
    was a street, a courtyard or open sky. A hole reads as missing data, which
    is true; a membrane reads as a wall that is not there, which is worse.
    """
    kept, dropped, kept_area, dropped_area = [], 0, 0.0, 0.0
    for group in mesh.groups:
        faces = np.asarray(group.faces).reshape(-1, 3)
        if not len(faces):
            continue
        areas = triangle_areas(mesh, faces)
        good = areas <= max_area
        dropped += int((~good).sum())
        dropped_area += float(areas[~good].sum())
        kept_area += float(areas[good].sum())
        if good.any():
            kept.append(Group(group.material, group.image, faces[good]))
    report = {
        "dropped_triangles": dropped,
        "dropped_area_m2": round(dropped_area, 1),
        "kept_area_m2": round(kept_area, 1),
        "dropped_area_fraction": round(dropped_area / (dropped_area + kept_area), 3)
        if (dropped_area + kept_area) else 0.0,
        "max_area_m2": max_area,
    }
    return Mesh(mesh.positions, mesh.uvs, kept, mesh.source), report


def crop(mesh: Mesh, lo, hi) -> Mesh:
    """Triangles whose centroid is inside the XY box [lo, hi].

    Centroid rather than any-vertex, so a triangle straddling the boundary
    belongs to exactly one crop and a seam does not double up.
    """
    lo = np.asarray(lo, dtype=float)[:2]
    hi = np.asarray(hi, dtype=float)[:2]
    keep_groups = []
    for group in mesh.groups:
        if not len(group.faces):
            continue
        centroids = mesh.positions[group.faces][:, :, :2].mean(axis=1)
        inside = np.all((centroids >= lo) & (centroids <= hi), axis=1)
        if inside.any():
            keep_groups.append(Group(group.material, group.image,
                                     group.faces[inside]))
    return Mesh(mesh.positions, mesh.uvs, keep_groups, mesh.source)
