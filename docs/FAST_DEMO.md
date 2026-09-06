# The fast demo

One demo. One page. One idea, shown rather than argued.

## The idea being shown

    a real place  ->  270 KB of seed  ->  a world you can walk, in any theme

Not "we reconstructed Amsterdam". The interesting claim is the *other* one: the
measured surface is thrown away on purpose, and what comes back is a coherent
Amsterdam-*like* street that a generator invented from a 270 KB description of
a 154 MB acquisition. A theme swap is then a lookup, not a recompile — which is
the whole point of the Spatial IR being theme-independent.

## The concept: one block, three worlds

A single self-contained HTML page. It opens on the Rembrandtplein block from
`AMSTERDAM.md` — the same 400 m of canal belt the compiler already produces —
rendered from the seed, walkable with WASD. Three buttons across the bottom:

    survey        victorian        neon

Pressing one re-materialises the *same* geometry through a different theme pack
in front of the viewer, in under a second. No reload, no second download: the
seed and the geometry are identical between the three, only the backend lookup
changed. That single second is the demo. Everything else on the page exists to
set it up.

Above the buttons, one line of text that never moves:

    Amsterdam, 400 m. 349 buildings from 3,659,659 measured returns.
    Seed 270 KB — 1002x smaller than the scan. None of these facades were seen.

The last sentence is not a disclaimer bolted on; it is the payload. Airborne
LiDAR never observed a single one of those walls, and saying so is what turns a
mediocre reconstruction into an honest and much more interesting result.

## Why this one and not another

- **Nothing new has to be measured.** `lidarworld compile ... --seed --gltf`
  already emits this block. The demo is packaging, not pipeline work.
- **It shows the invariant, which nothing else does.** Every current artifact
  shows a place. None shows that the place is theme-independent, and that is
  the part that is architecturally hard and easy to disbelieve.
- **It survives our worst numbers.** Forward validation at ~28% and trees at
  6.8x are both statements about fidelity to the scan. This demo is not
  claiming fidelity to the scan. The weak numbers are off the critical path
  rather than hidden.
- **Precedent exists.** `tools/build_walk_demo.py` already inlines a mesh into
  one HTML file for a host that blocks external fetches. Same shape, new
  content.

## Build

Roughly a day, in this order, each step something to look at.

1. `lidarworld compile` the block three times, once per theme, `--gltf`.
   Check the three GLBs differ only in materials. (~1 h)
2. `tools/glb_shot.py` each one. **Look at them.** If victorian and neon read
   as the same grey street, the demo has no content and the rest is wasted —
   this is the go/no-go. (~1 h)
3. Merge to one page: geometry once, three material sets, a button row that
   rebinds them. (~3 h)
4. Crop to the byte budget. 16 MB page, base64 costs a third, so ~11 MB of
   payload for geometry plus three material sets. Expect to cut the block from
   400 m to something like 150 m; cut extent before cutting texture resolution,
   because a short street that looks right beats a long one that looks cheap.
   (~2 h)
5. `tools/shoot.py` the finished page from the spawn point and two street
   corners, in all three themes. Nine stills. Ship only if all nine hold. (~1 h)

## What it is not

- Not a level. There is no objective, no interaction beyond walking and
  swapping.
- Not real-time generation. The three themes are baked at build time; the
  runtime swap is a material rebind, which is honest — the *compile* is the
  lookup, and the page is showing its output.
- Not Helsinki. Kalasatama has the better surface and no seed pipeline; this
  demo is about the seed. Keep them apart.

## How it fails

- **The three themes look alike.** The most likely failure by a distance.
  Detected at step 2, before any page work. Fix by widening the packs
  (silhouette and proportion, not just colour), not by narrowing the demo.
- **The seed-built street reads as a grey box city** and the theme swap has
  nothing to act on. Then the demo is early and the honest move is to say so,
  not to dress it with fog.
- **The byte budget eats the block** down to a length that reads as a corridor.
  Fall back to one canal frontage seen from across the water, which is the
  view Amsterdam is recognisable from anyway.
- **The render flatters or flattens.** `glb_shot.py` point-samples with no
  mipmap and a flat Lambert; masonry moirés and relief disappears. Judge
  materials in the browser page, not in the software render.
