"""Data structures for CGARF"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import time


class EntityType(Enum):
    """Types of code entities"""
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    GLOBAL_VAR = "global_variable"
    CONFIG = "config"
    API_CALL = "api_call"


class EdgeType(Enum):
    """Types of edges in code graph"""
    CONTAINMENT = "containment"  # Parent-child relationships
    REFERENCE = "reference"      # Usage and call relationships
    DEPENDENCY = "dependency"    # Import/dependency


@dataclass
class CodeEntity:
    """Represents a code entity (file, class, function, etc.)"""
    entity_id: str              # Unique ID: "file::class::function"
    file_path: str
    class_name: Optional[str] = None
    function_name: Optional[str] = None
    variable_name: Optional[str] = None
    entity_type: EntityType = EntityType.FILE
    code_snippet: str = ""
    line_range: Tuple[int, int] = (0, 0)
    semantic_summary: Optional[str] = None

    def __hash__(self):
        return hash(self.entity_id)

    def __eq__(self, other):
        if isinstance(other, CodeEntity):
            return self.entity_id == other.entity_id
        return False


@dataclass
class CodeEdge:
    """Represents an edge in code graph"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    semantic_info: Optional[str] = None

    def __hash__(self):
        return hash((self.source_id, self.target_id, self.edge_type))

    def __eq__(self, other):
        if isinstance(other, CodeEdge):
            return (
                self.source_id == other.source_id and
                self.target_id == other.target_id and
                self.edge_type == other.edge_type
            )
        return False


@dataclass
class CRGNode(CodeEntity):
    """Node in Causal Relevance Graph"""
    credibility: float = 0.0
    initial_strength: float = 0.0


@dataclass
class CRGEdge(CodeEdge):
    """Edge in Causal Relevance Graph with strength"""
    strength: float = 0.0  # [0, 1]
    initial_strength: float = 0.0
    semantic_contribution: Optional[str] = None


@dataclass
class PathEvidence:
    """Represents a path from candidate to failure anchor"""
    nodes: List[CRGNode] = field(default_factory=list)
    edges: List[CRGEdge] = field(default_factory=list)
    failure_anchor: str = ""
    path_credibility: float = 0.0
    path_string: str = ""  # Human-readable format
    syntax_score: float = 0.0
    
    def __hash__(self):
        return hash(self.path_string)


@dataclass
class PatchCandidate:
    """Represents a candidate patch"""
    patch_id: str
    location: str  # entity_id of code location
    patch_content: str  # Search/Replace format
    generated_round: int
    credibility_from_location: float
    final_score: float = 0.0
    test_result: Optional[str] = None  # PASS/FAIL/REGRESSION_FAIL
    
    def __hash__(self):
        return hash(self.patch_id)


@dataclass
class ReflectionScore:
    """Self-reflection evaluation score for a patch"""
    semantic_consistency: float  # [0, 1]
    causal_alignment: float      # [0, 1]
    minimal_edit: float          # [0, 1]
    combined_score: float
    reasoning: Dict[str, str] = field(default_factory=dict)
    next_suggestion: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class PatchBatch:
    """Batch of patches for a single candidate location"""
    location: str
    candidates: List[PatchCandidate] = field(default_factory=list)
    reflection_scores: List[ReflectionScore] = field(default_factory=list)
    consensus_pattern: Optional[str] = None
    distillation_scores: List[float] = field(default_factory=list)
    top_patches: List[PatchCandidate] = field(default_factory=list)
    
    def __len__(self):
        return len(self.candidates)


@dataclass
class IssueContext:
    """Represents an issue to be repaired"""
    id: str
    description: str
    repo_path: str
    candidates: List[str]  # entity_ids
    test_framework: str = "pytest"
    timeout_seconds: int = 120
    metadata: Dict = field(default_factory=dict)


@dataclass
class RepairResult:
    """Result of a repair attempt"""
    issue_id: str
    patch: Optional[PatchCandidate] = None
    success: bool = False
    confidence: float = 0.0
    reasoning_chain: str = ""
    metrics: Dict = field(default_factory=dict)


@dataclass
class RepairCandidate:
    """Represents a repair candidate (patched code)"""
    id: str
    original_code: str
    repaired_code: str
    mutation_type: str
    affected_lines: List[int] = field(default_factory=list)
    confidence: float = 0.0
    semantic_summary: str = ""
    test_results: Optional[Dict[str, Any]] = None
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, RepairCandidate):
            return False
        return self.id == other.id


@dataclass
class VerifiedPatch:
    """A patch that has been verified through testing"""
    repair: 'RepairCandidate'
    test_results: Optional[Dict[str, Any]] = None
    verification_score: float = 0.0
    confidence: float = 0.0
    pass_rate: float = 0.0


# Type aliases for convenience
CandidateList = List[str]
PatchList = List[PatchCandidate]
EdgeWeightDict = Dict[str, float]
DebateResult = Dict[str, any]
