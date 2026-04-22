"""CGARF - Causality-Guided Automated Program Repair Framework"""

__version__ = "0.1.0"
__author__ = "CGARF Contributors"

from .common.data_structures import (
    CodeEntity,
    CodeEdge,
    CRGNode,
    CRGEdge,
    PathEvidence,
    PatchCandidate,
    ReflectionScore,
    PatchBatch,
    IssueContext,
    RepairResult,
    RepairCandidate,
    VerifiedPatch,
)

__all__ = [
    "CodeEntity",
    "CodeEdge",
    "CRGNode",
    "CRGEdge",
    "PathEvidence",
    "PatchCandidate",
    "ReflectionScore",
    "PatchBatch",
    "IssueContext",
    "RepairResult",
    "RepairCandidate",
    "VerifiedPatch",
]
