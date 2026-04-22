"""
Phase 1: Causal Analysis - CG-MAD Multi-Agent Debate Framework
==============================================================

Core modules for building causal relevance graphs and multi-agent debate.

Main Components:
1. causal_relevance_graph: Core CRG construction
   - CodeGraph: Code structure representation
   - CausalRelevanceGraph: Issue-conditioned graph
   - CRGBuilder: Orchestrates construction
   - PathCollector: Collects candidate paths

2. code_graph_builder: Automatic code graph extraction
   - ASTAnalyzer: Python AST analysis
   - CodeGraphBuilder: Full repository analysis

3. cg_mad: Phase 2 multi-agent causal debate and dynamic optimization
   - CGMADMechanism: paper-aligned path/location debate engine
"""

from .causal_relevance_graph import (
    # Enums
    EntityType,
    RelationType,
    
    # Data Classes
    CodeEntity,
    CodeRelation,
    FailureEvidence,
    CRGEdge,
    
    # Core Classes
    CodeGraph,
    CausalRelevanceGraph,
    EdgeWeightingStrategy,
    DataFlowWeighting,
    CRGBuilder,
    PathCollector,
)

from .code_graph_builder import (
    ASTAnalyzer,
    CodeGraphBuilder,
)

from .cg_mad import (
    CGMADMechanism,
    CGMADResult,
    CandidateAssessment,
    PathSummary,
)

__all__ = [
    # Enums
    "EntityType",
    "RelationType",
    
    # Data Classes
    "CodeEntity",
    "CodeRelation",
    "FailureEvidence",
    "CRGEdge",
    
    # Core Classes
    "CodeGraph",
    "CausalRelevanceGraph",
    "EdgeWeightingStrategy",
    "DataFlowWeighting",
    "CRGBuilder",
    "PathCollector",
    
    # Builder
    "ASTAnalyzer",
    "CodeGraphBuilder",

    # Phase 2
    "CGMADMechanism",
    "CGMADResult",
    "CandidateAssessment",
    "PathSummary",
]
