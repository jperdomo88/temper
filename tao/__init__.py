"""
TAO — A Behavioral Audit Interface for Agentic AI.

Reference validator for TAO v0.11. Implements:
    - Tuple structural validation (against tao_tuple.schema.json)
    - Placeholder rejection on entity identifiers
    - Semantic-mechanical mapping enforcement (Appendix B)
    - Justification requirements (flagged verbs, harm acknowledgment)
    - Claim-Check Delta with the five-class teleological taxonomy
    - Mission Profile override discipline (structural checks)

Not yet implemented (planned for v0.12+):
    - RFC 8785 (JCS) canonical serialization and signature verification
    - Authority-chain resolution against an attested registry
    - Profile signature verification

Conformance with the spec is tested by running the bundled test vector suite:
    $ tao check-suite spec/test_vectors.json
"""

__version__ = "0.11.3"

from .schema import validate_tuple_schema, validate_profile_schema
from .mapping import (
    load_mappings,
    apply_mapping_rules,
    apply_override_discipline,
    MappingResult,
)
from .justification import check_justification, JustificationResult
from .ccd import claim_check_delta, CCDResult, TeleologicalClass
from .validator import validate_tuple, ValidationResult

__all__ = [
    "__version__",
    "validate_tuple",
    "ValidationResult",
    "validate_tuple_schema",
    "validate_profile_schema",
    "load_mappings",
    "apply_mapping_rules",
    "apply_override_discipline",
    "MappingResult",
    "check_justification",
    "JustificationResult",
    "claim_check_delta",
    "CCDResult",
    "TeleologicalClass",
]
