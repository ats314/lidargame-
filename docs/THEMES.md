# Theme packs

A theme is data, not code: a list of rules over `(role, context)` and a table of
material specs. Nothing in a pack references a vertex, and nothing in the
geometry references a pack.

## Anatomy

```jsonc
{
  "id": "victorian",
  "name": "Victorian Brick",
  "era": "1850-1900",
  "fallback": "default",
  "environment": { "sky": [...], "fog": [...], "fogDensity": 0.01, "sunDirection": [...] },
  "materials": [
    { "id": "stock_brick", "generator": "brick",
      "baseColor": [0.56, 0.30, 0.22],
      "params": { "cols": 6, "rows": 14, "wear": 0.35 },
      "roughness": 0.9, "scale": 2.4, "license": "Proprietary - All Rights Reserved",
      "era": ["1850-1900"], "tags": ["wall", "brick"] }
  ],
  "rules": [
    { "material": "portland_stone", "role": "surface.wall",
      "ctx_any": ["corner_convex", "corner_concave"], "priority": 3,
      "note": "quoins on building corners" },
    { "material": "stock_brick", "role": "surface.wall" },
    { "material": "default", "role": "*" }
  ]
}
```

`scale` is **world metres per texture tile**, so a material keeps its real-world
size on any surface regardless of how the mesh was cut.

## Rule matching

A rule matches when all of these hold:

- `role` is a prefix of the request's role on a dot boundary — `surface.wall`
  matches `surface.wall.vertical` but not `surface.walltop`. `*` matches all.
- every flag in `ctx_all` is set
- at least one flag in `ctx_any` is set (if non-empty)
- no flag in `ctx_none` is set
- `semantic` matches, if given

Rules are evaluated **most specific first**: sorted by `priority` descending,
then by specificity (required flags count double, a named role adds two, each
level of role depth adds one), then by declaration order. A pack should end with
a `"role": "*"` catch-all; `lidarworld themes -v` and the test suite both check
that every role resolves.

Trace a decision:

```bash
lidarworld explain --theme victorian \
  --role surface.wall.vertical --context corner_convex,street_facing
```

The viewer's inspector shows the same thing for whatever is under the crosshair,
including the rule's `note`.

## Writing a pack

Start from `src/lidarworld/themes/packs/survey.json` — it is the diagnostic
pack, so its rules map one-to-one onto the context flags and it is the clearest
template. Then:

```bash
lidarworld compile tile.las -o build/w --theme ./my_pack.json
```

Packs load by built-in id or by path. `ThemePack.validate()` runs on load and
reports unknown materials, unknown context flags and a missing fallback all at
once rather than failing on the first.

## Material resolution backends

A `MaterialSpec` says what a material *is*; a resolver decides who provides it.

- **`procedural`** (default) — synthesised from code at bake time. No third-party asset
  licensing, tiny repo, and a new era is a parameter block. Generators live in
  `themes/procedural.py`: `brick`, `stone_block`, `plaster`, `concrete`,
  `asphalt`, `cobble`, `roof_tile`, `metal_panel`, `glass`, `wood_plank`,
  `foliage`, `grass`, `gravel`, `water`, `neon_panel`, `thatch`. All are
  tileable — the noise lattice wraps.
- **`image`** — a CC0 library, a photogrammetry capture or your own pack.
  Override by material id; the geometry never changes.
- **`engine`** — pass a name through to an engine-native material.

Every spec carries `license`, `source`, `scale`, `era` and `tags`, and those
travel into `theme.json`, into glTF `extras`, and into the viewer's inspector.

## Adding a procedural generator

A generator takes keyword params and returns `(albedo, height, roughness)` as
float arrays in `[0, 1]`:

```python
def corrugated(seed=0, color=(0.5, 0.5, 0.55), ribs=16, **_):
    x = np.arange(SIZE)[None, :] / SIZE * ribs
    profile = 0.5 + 0.5 * np.sin(x * 2 * np.pi)
    grain = fbm(SIZE, seed, 4, 8)
    albedo = _tint(grain * 0.3, color, grain, 0.12) * (0.7 + 0.5 * profile)[:, :, None]
    return np.clip(albedo, 0, 1), profile, np.full((SIZE, SIZE), 0.4)

GENERATORS["corrugated"] = corrugated
```

Use `ribs` values that divide the texture size, or the wrap will seam. The test
suite checks every registered generator for tileability and channel shape.

## What a theme cannot do

It cannot move a vertex, add a hole, or change what a surface *is*. If a wall
should have a window, that is a reconstruction question, not a theme question.
This boundary is what makes runtime re-skinning free.
