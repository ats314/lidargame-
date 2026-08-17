"""Static checks on the viewer's GLSL, because nothing else looks at it.

The viewer has no test runner and CI never compiles a shader, so a broken one
is found by a human opening the page -- or not found at all, since a varying
that is declared but never written is valid GLSL that silently reads zero.

That is not hypothetical: adding height-attenuated fog put `vWorldZ = ...` in
the surface vertex shader while declaring `out float vWorldZ` in the point one.
Both files still parsed. The points would have faded against a height of zero.

These are text checks, not a compiler. They catch the wiring mistakes -- a
varying written in the wrong stage, a uniform the renderer never sets -- and
say nothing about whether the GLSL is correct.
"""
import re
from pathlib import Path

import pytest

VIEWER = Path(__file__).resolve().parents[1] / "viewer" / "src"
SHADERS = (VIEWER / "shaders.js").read_text()
RENDER = (VIEWER / "render.js").read_text()

#: Vertex/fragment pairs that are linked into one program by render.js.
PROGRAMS = [("SURFACE_VS", "SURFACE_FS"), ("POINTS_VS", "POINTS_FS"),
            ("INSTANCE_VS", "INSTANCE_FS"), ("SKY_VS", "SKY_FS")]


def _source(name: str) -> str:
    match = re.search(rf"export const {name} = /\* glsl \*/`(.*?)`;", SHADERS, re.S)
    assert match, f"no shader named {name}"
    return match.group(1)


def _declared(source: str, direction: str) -> set[str]:
    return set(re.findall(rf"^\s*(?:flat\s+)?{direction}\s+\w+\s+(\w+)\s*;",
                          source, re.M))


def _assigned(source: str) -> set[str]:
    """Names assigned anywhere, including inside a branch.

    `if (uColorMode == 0) vColor = ...` is a real assignment, so anchoring this
    to the start of a line reports a varying as unwritten when it is written
    three times. The lookarounds keep `==`, `<=` and friends out.
    """
    return set(re.findall(r"(?<![=!<>+\-*/])\b(\w+)\s*=(?!=)", source))


@pytest.mark.parametrize("vs,fs", PROGRAMS)
def test_every_varying_read_is_written(vs, fs):
    """A fragment `in` with no matching vertex `out` reads zero, silently."""
    vertex, fragment = _source(vs), _source(fs)
    outs, ins = _declared(vertex, "out"), _declared(fragment, "in")
    missing = ins - outs
    assert not missing, f"{fs} reads {sorted(missing)}, which {vs} does not declare"

    never_written = outs - _assigned(vertex)
    assert not never_written, (
        f"{vs} declares {sorted(never_written)} but never assigns it")


@pytest.mark.parametrize("name", [n for pair in PROGRAMS for n in pair])
def test_shared_chunks_are_interpolated_not_pasted(name):
    """Each shader gets the shared source by reference, so there is one copy."""
    source = _source(name)
    for call, chunk in (("fogFactor(", "${FOG}"), ("shade(", "${LIGHTING}")):
        if call in source and f"float {call[:-1]}" not in source:
            assert chunk in source, f"{name} calls {call} without {chunk}"


def test_fog_uniforms_are_set_by_the_renderer():
    """A uniform nobody sets is zero: scaleHeight 0 divides by zero."""
    for name in ("SURFACE_FS", "POINTS_FS", "INSTANCE_FS"):
        for uniform in _declared_uniforms(_source(name)) & {
                "uFogBase", "uFogHeight", "uFogDensity", "uFogColor"}:
            assert f"uniforms.{uniform}" in RENDER, (
                f"{name} declares {uniform} and render.js never sets it")


def _declared_uniforms(source: str) -> set[str]:
    return set(re.findall(r"^\s*uniform\s+\w+\s+(\w+)", source, re.M))


def test_fog_is_height_attenuated_everywhere_or_nowhere():
    """Surfaces, instance proxies and points must share an atmosphere, or the
    point cloud floats in front of the buildings it was meshed from."""
    for name in ("SURFACE_FS", "POINTS_FS", "INSTANCE_FS"):
        source = _source(name)
        assert "fogFactor(" in source, f"{name} does not use the shared fog"
        assert "exp(-vDistance * uFogDensity)" not in source, (
            f"{name} still has the old uniform-density fog inline")


def test_the_fog_layer_is_anchored_to_the_world_not_the_camera():
    """Anchoring to the camera makes distant blocks breathe as you fly."""
    assert "this.fogBase" in RENDER
    assert re.search(r"fogBase\s*=\s*world\.bounds", RENDER), (
        "fog base should come from the world's floor")
