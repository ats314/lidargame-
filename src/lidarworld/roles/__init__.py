"""Geometric role taxonomy and per-point role assignment."""
from .classify import classify, role_histogram, THRESHOLDS  # noqa: F401
from .taxonomy import (Ctx, ROLE_IDS, ROLE_INDEX, ROLES, Role, role_matches,  # noqa: F401
                       role_or_unknown)

__all__ = ["ROLES", "ROLE_IDS", "ROLE_INDEX", "Role", "Ctx", "role_matches",
           "role_or_unknown", "classify", "role_histogram", "THRESHOLDS"]
