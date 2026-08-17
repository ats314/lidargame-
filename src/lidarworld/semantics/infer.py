"""Semantic inference for clouds that arrive without labels.

Public airborne tiles usually ship ASPRS classes and SemanticKITTI ships
per-point labels, and when they do we use them -- a published classification
beats anything inferred here. This module is the fallback for the raw scans most
people actually have: PCD off a robot, PLY out of CloudCompare, a bare KITTI
``.bin``.

It is deliberately a transparent rule cascade over height-above-ground and the
multiscale descriptors, not a learned model. Every decision is inspectable, it
needs no weights to ship, and it writes an honest per-point confidence that
propagates into node confidence in the world graph. Where a real segmentation
network is available (RandLA-Net, KPConv, SPT), its output can be dropped in as
a `semantic` channel and this stage skipped entirely -- see docs/RESOURCES.md.
"""
from __future__ import annotations

import numpy as np

from ..types import SEMANTIC_INDEX, PointCloud

S = SEMANTIC_INDEX


#: A car is about 8 m2 and roughly the shape a kerb, a planter or a loading pad
#: also makes from above. Separating them needs enough returns to resolve the
#: 1.4 m step up to a roof, which airborne acquisitions do not have: measured
#: over the Denver block, `density` in the vehicle height band peaks at 0.98
#: against the 1.0 the original rule demanded.
#:
#: The threshold below is the original one. What changed is that failing it is
#: now a deliberate refusal with a measurement behind it, rather than a
#: comparison that silently could not succeed.
VEHICLE_MIN_DENSITY = 1.0


def _density_supports_vehicles(density, hag) -> bool:
    """Whether this cloud is dense enough for vehicle inference to mean anything.

    Returns False for airborne acquisitions, which is the honest answer: at
    3.6 pts/m2 the candidate set is 6% cars and 94% street furniture, and a
    wrong permanent object is harder to notice than an absent one.
    """
    import numpy as _np

    band = (hag > 0.3) & (hag < 2.6)
    if not band.any():
        return False
    return bool(_np.percentile(density[band], 90) >= VEHICLE_MIN_DENSITY)


def infer(cloud: PointCloud, *, overwrite: bool = False) -> PointCloud:
    """Fill in `semantic` (and `semantic_confidence`) from geometry."""
    cloud.require("hag", "planarity", "sphericity", "linearity", "verticality", "density")
    n = len(cloud)

    existing = cloud.get("semantic")
    # USGS 3DEP tiles routinely classify only ground and noise and leave
    # everything else as ASPRS class 1, so a majority test discards a perfectly
    # good ground classification. Any meaningful labelled minority is worth
    # keeping -- inference then fills only the unlabelled remainder.
    have_labels = existing is not None and (existing != 0).mean() > 0.02
    gap = None
    if have_labels and not overwrite:
        # Trust the published classification, but do not leave the leftovers on
        # the floor: real tiles carry a meaningful "unclassified" fraction
        # (street furniture, vehicles, anything the vendor's filter skipped),
        # and those points still have to become something.
        gap = existing == 0
        if gap.sum() < max(32, 0.002 * n):
            if "semantic_confidence" not in cloud:
                cloud["semantic_confidence"] = np.where(existing != 0, 0.95, 0.3).astype(np.float32)
            cloud.meta["semantic_source"] = "source labels"
            return cloud

    hag = cloud["hag"]
    planarity = cloud["planarity"]
    sphericity = cloud["sphericity"]
    linearity = cloud["linearity"]
    verticality = cloud["verticality"]
    density = cloud["density"]
    intensity = cloud.get("intensity", np.full(n, 0.3, dtype=np.float32))

    semantic = np.full(n, S["unclassified"], dtype=np.uint8)
    confidence = np.full(n, 0.35, dtype=np.float32)

    def assign(mask, cls, conf):
        fresh = mask & (semantic == S["unclassified"])
        semantic[fresh] = S[cls]
        confidence[fresh] = conf

    # Sparse returns far above everything: birds, dust, multipath.
    assign((density < 0.05) & (hag > 8.0), "noise", 0.5)

    # Terrain first -- everything else is defined relative to it.
    ground = hag < 0.25
    flat = planarity > 0.35
    # Asphalt is a poor 905 nm reflector; bare earth and grass return more.
    paved = ground & flat & (intensity < 0.28)
    assign(paved, "road", 0.55)
    assign(ground, "ground", 0.8)

    # Water: flat, very low return, and it sits at a locally constant level.
    assign((hag > -0.6) & (hag < 0.35) & (intensity < 0.05) & (planarity > 0.5), "water", 0.5)

    # Thin, tall, linear -> poles and wires.
    thin = (linearity > 0.65) & (density < 3.0)
    assign(thin & (hag > 4.0) & (verticality < 0.35), "wire", 0.55)
    assign(thin & (hag > 1.5) & (verticality > 0.7), "pole", 0.6)

    # Structure: planar and either upright or clearly above the ground.
    wall = (verticality > 0.65) & (planarity > 0.45) & (hag > 1.5)
    roof = (verticality < 0.35) & (planarity > 0.5) & (hag > 2.5)
    assign(wall | roof, "building", 0.7)

    # Scattered volume -> vegetation. Shape alone is a bad test: at airborne
    # densities a noisy roof or a parapet is "scattered" too, which is how a
    # 74 m high-rise ends up labelled a tree. Return structure separates them --
    # a pulse through a canopy comes back several times, one off a roof comes
    # back once -- so where the sensor recorded it, it is the primary evidence
    # and shape only has to agree weakly.
    scattered = sphericity > 0.13
    num_returns = cloud.get("num_returns")
    if num_returns is not None and int(np.max(num_returns, initial=0)) > 1:
        penetrated = num_returns > 1
        # Measured on the Denver 3DEP tile, confident roof vs confident scatter:
        # multi-return 9.3% vs 70.7%. Canopy is where the pulse got through.
        canopy = scattered & penetrated
        assign(canopy & (hag > 2.0), "vegetation_high", 0.75)
        assign(canopy & (hag > 0.35), "vegetation_low", 0.65)
        # Single-return scatter high up is building clutter -- parapets, plant
        # rooms, rooftop HVAC -- not a tree. This is the branch that matters:
        # letting it fall through to vegetation is what turns a 74 m high-rise
        # into a 74 m tree.
        assign(scattered & (hag > 2.5), "building", 0.45)
        assign(scattered & (hag > 0.35), "vegetation_low", 0.4)
    else:
        assign(scattered & (hag > 2.0), "vegetation_high", 0.65)
        assign(scattered & (hag > 0.35), "vegetation_low", 0.6)

    # Compact, low, semi-planar blobs sitting on the ground -> vehicles.
    #
    # This rule required `density > 1.0` and therefore could never fire on
    # airborne data: `density` is count/(scale^3 * 27), and over the Denver
    # block the *maximum* in this height band is 0.98. That is not a tuning
    # problem, it is a structurally unsatisfiable test -- the threshold came
    # from terrestrial scans at hundreds of points per square metre, where it
    # is fine, and nothing said so.
    #
    # Removing it does not produce vehicles. Measured on the LoDo block: the
    # planar, unscattered, 0.3-2.6 m candidates cluster into 451 objects of
    # which 26 are car-shaped -- 6% precision, and tightening the height band
    # does not help (5% at hag>0.8, 5% at hag>1.0, 2% at hag>1.2). The
    # candidates are kerbs, planters, steps and loading pads: median cluster
    # 5.6 x 4.9 m and 0.6 m tall, which is not a car.
    #
    # So the honest output is no vehicles, and the difference that matters is
    # that it is now a refusal rather than an accident. At 3.6 pts/m2 a car is
    # ~24 returns spread over a shape a kerb also makes, and enabling this
    # would fossilise 94% false positives into the static world -- which is
    # worse than missing them, because a wrong permanent object is harder to
    # notice than an absent one.
    #
    # What would change the answer, in order of cost: gating candidates on the
    # acquired pavement and parking polygons (a first pass put 63 of 83
    # car-sized clusters inside one, so the context does discriminate);
    # mobile or terrestrial LiDAR, where the original density test works as
    # written; or imagery.
    if _density_supports_vehicles(density, hag):
        assign((hag > 0.3) & (hag < 2.6) & (planarity > 0.3) & ~scattered,
               "vehicle", 0.4)

    # Low remaining structure is most likely fencing.
    assign((hag > 0.3) & (hag < 2.5) & (verticality > 0.6), "fence", 0.4)

    assign(hag < 1.5, "ground", 0.4)

    if gap is not None:
        # Inference only fills the holes; published labels win everywhere else.
        kept = existing.copy()
        kept[gap] = semantic[gap]
        conf = np.full(n, 0.95, dtype=np.float32)
        conf[gap] = confidence[gap] * 0.8
        cloud["semantic"] = kept
        cloud["semantic_confidence"] = conf
        cloud.meta["semantic_source"] = (
            f"source labels + inference for {int(gap.sum()):,} unclassified points")
        return cloud

    cloud["semantic"] = semantic
    cloud["semantic_confidence"] = confidence
    cloud.meta["semantic_source"] = "inferred (rule cascade)"
    return cloud


def class_histogram(cloud: PointCloud) -> dict[str, int]:
    from ..types import SEMANTIC_CLASSES
    semantic = cloud.get("semantic")
    if semantic is None:
        return {}
    counts = np.bincount(semantic, minlength=len(SEMANTIC_CLASSES))
    return {name: int(c) for name, c in zip(SEMANTIC_CLASSES, counts) if c}
