"""World-building: gap classification, repair records, resolved world seed.

The compiler's inverse half asks what the returns support. This half asks what
a coherent world requires, and -- the part that keeps the two honest -- writes
down which of the two answered for every piece of the result.
"""
from . import gaps
from .records import (BOUNDARY_TYPES, COMPLETION_TIERS, EPISTEMIC_STATES,
                      GAP_TYPES, REPAIR_PASSES, BoundarySeed, GapRecord,
                      RepairLog, RepairRecord)

__all__ = ["BoundarySeed", "GapRecord", "RepairRecord", "RepairLog", "gaps",
           "GAP_TYPES", "REPAIR_PASSES", "BOUNDARY_TYPES", "EPISTEMIC_STATES",
           "COMPLETION_TIERS"]
