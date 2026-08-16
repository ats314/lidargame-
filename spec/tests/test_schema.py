from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sir.reference import validate_document


def test_reference_scene_validates():
    validate_document(
        ROOT / "examples" / "reference_scene.sir.json",
        ROOT / "schema" / "spatial_ir_v0_1.schema.json",
    )
