"""
CRG Construction Documentation
=============================

Comprehensive guide to Phase 1.1 Causal Relevance Graph Construction

Reference: CGARF Paper Section 3.1.1
"因果相关图构建（CRG Construction）"
"""

# ============================================================================
# CORE CONCEPTS
# ============================================================================

"""
1. CODE GRAPH: Foundation Structure
═════════════════════════════════════════════════════════════════════════════

The code graph G_CG = (V_CG, ε_CG) represents the program's static structure:

V_CG (Vertices - Code Entities):
  • Files         - Source code files
  • Classes       - Class definitions
  • Methods       - Methods and functions
  • Variables     - Local and member variables
  • Parameters    - Function parameters
  • Imports       - Imported modules

ε_CG (Edges - Structural Relations):
  
  1. CONTAINMENT Relations (包含):
     - File contains Class
     - Class contains Method
     - Method contains Variable
     
     These form the static hierarchy of the code.
  
  2. REFERENCE Relations (引用):
     - Calls:      Method calls another method
     - References: Uses a variable/function/class
     - Inherits:   Class inherits from parent class
     - Reads:      Reads a variable
     - Writes:     Writes/assigns to a variable
     
     These represent dynamic execution dependencies.


2. CAUSAL RELEVANCE GRAPH: Issue-Conditioned
═════════════════════════════════════════════════════════════════════════════

Key insight: CRG ≠ PDG / call-graph / def-use-graph

The CRG is specifically conditioned on FAILURE EVIDENCE:
  • Only includes edges relevant to explaining the current bug
  • Weights edges by relevance to failure manifestation
  • Direction: Symptom ← (causal influence) ← Root Cause

G_CRG = (V_CRG, ε_CRG, C)

V_CRG: Same as code graph nodes
ε_CRG: Causal edges (subset of code graph)
C:     Edge weights c_i ∈ [0,1] by relevance

Weight Interpretation:
  • c_i = 1.0  : Direct causal contributor to failure
  • c_i = 0.5  : Possible indirect influence
  • c_i = 0.1  : Weak or no relevance


3. FAILURE EVIDENCE: The Conditioning Factor
═════════════════════════════════════════════════════════════════════════════

Failure evidences define WHAT symptom to explain:

FailureEvidence contains:
  • symptom_type:     "assertion_failure", "exception", "wrong_output"
  • symptom_location: Where symptom manifests (test file, line)
  • symptom_message:  Error description
  • stack_trace:      Call chain leading to failure
  • test_case_id:     Which test case exposed the bug

Example:
  AssertionError in test_add[case1]:
  Expected: 5
  Got:      6
  Stack: test_file.py→calc.add()→... (line 14)


4. PATH COLLECTION: Explaining the Bug
═════════════════════════════════════════════════════════════════════════════

For target symptom location d, collect paths Π = {π₁, π₂, ..., πₘ}

Each path π_i represents a potential causal explanation:
  Root_Cause₁ → ... → Intermediate → ... → Symptom

Path ranking uses edge weights:
  Path_Score = ∏(c_i) for all edges in path

Higher scoring paths are more likely root causes.


5. CONSTRUCTION WORKFLOW
═════════════════════════════════════════════════════════════════════════════

Step 1: SYMPTOM IDENTIFICATION
  ✓ Parse failure evidence(s)
  ✓ Identify code entities mentioned in stack trace
  ✓ Mark them as symptom nodes

Step 2: BACKWARD TRACING
  ✓ From each symptom node, trace backwards in code graph
  ✓ Find all entities that can influence it
  ✓ Use DFS to traverse dependency chains

Step 3: EDGE WEIGHTING
  ✓ For each potential causal edge:
    - Compute base weight by relation type
    - Adjust by failure evidence proximity
    - Normalize to [0, 1]

Step 4: PATH COLLECTION
  ✓ Find all paths from potential root causes to symptoms
  ✓ Prune low-relevance paths (c_i < threshold)
  ✓ Rank paths by cumulative weight

Step 5: OUTPUT
  ✓ Candidate set C of root cause locations
  ✓ Supporting evidence (causal paths)
  ✓ Confidence scores for each candidate
"""

# ============================================================================
# API REFERENCE
# ============================================================================

"""
CLASS: CodeEntity
─────────────────

Represents a code element (file, class, method, variable, etc.)

Attributes:
  id           (str):            Unique identifier (e.g., "file.py::ClassName::method")
  name         (str):            Display name
  entity_type  (EntityType):     FILE, CLASS, FUNCTION, VARIABLE, PARAMETER, IMPORT
  file_path    (str):            Source file path
  line_start   (Optional[int]):  Starting line number
  line_end     (Optional[int]):  Ending line number
  parent_id    (Optional[str]):  Parent entity ID (for hierarchy)

Methods:
  __hash__() → HashableEntity    : Make hashable for use in sets/dicts
  __eq__()   → bool              : Equality comparison by ID


CLASS: CodeGraph
────────────────

Container for all code structure (V_CG, ε_CG)

Methods:
  add_entity(entity: CodeEntity) → None
    Add a code entity to the graph
  
  add_relation(relation: CodeRelation) → None
    Add a structural relation
  
  get_entity(entity_id: str) → Optional[CodeEntity]
    Retrieve entity by ID
  
  get_children(entity_id: str) → List[CodeEntity]
    Get entities contained by this one (containment relations)
  
  get_parents(entity_id: str) → List[CodeEntity]
    Get entities that contain this one
  
  get_references(entity_id: str) → List[Tuple[CodeEntity, RelationType]]
    Get all entities referenced by this one

Properties:
  entities    : Dict[str, CodeEntity]  - All entities by ID
  relations   : List[CodeRelation]     - All relations
  nx_graph    : nx.DiGraph             - NetworkX representation


CLASS: CausalRelevanceGraph
────────────────────────────

Issue-conditioned CRG with weighted edges

Methods:
  add_edge(edge: CRGEdge) → None
    Add a weighted causal edge
  
  get_edge_weight(source_id, target_id) → Optional[float]
    Get weight of edge (None if not exists)
  
  get_ancestors(entity_id, max_depth) → List[str]
    Get all potential root causes (backward reach)
  
  get_descendants(entity_id, max_depth) → List[str]
    Get all potential failures (forward reach)

Properties:
  code_graph       : CodeGraph
  failure_evidences: List[FailureEvidence]
  edges            : Dict[Tuple[str,str], CRGEdge]
  node_weights     : Dict[str, float]


CLASS: CRGBuilder
──────────────────

Orchestrates CRG construction from code graph and failures

Methods:
  __init__(code_graph, weighting_strategy)
    Initialize with base graph and strategy
  
  build(failure_evidences: List[FailureEvidence]) → CausalRelevanceGraph
    Build complete CRG
    
    Process:
    1. Identify symptom entities from failures
    2. Trace causal paths backward
    3. Weight edges by relevance
    4. Return weighted graph

  _identify_symptom_entities(failures) → List[CodeEntity]
    Map failure locations to code entities
  
  _trace_causal_paths(symptom_entity, crg, failures, max_depth)
    DFS backward tracing to find root causes


CLASS: PathCollector
─────────────────────

Collects candidate paths in CRG

Methods:
  collect_paths(target_id, max_path_length, max_paths) → List[List[str]]
    Collect all paths from root causes to target
    
    Returns:
      List of paths, where each path is [cause₁, ..., symptom]
      Paths ordered by relevance (highest weight first)

  Results:
    - Empty list if no paths found (isolated nodes)
    - Multiple paths if multiple root causes
    - Each path is candidate explanation


CLASS: CodeGraphBuilder
────────────────────────

Automatic extraction from source code

Methods:
  build_from_repository(repo_path: str) → CodeGraph
    Analyze entire repository
    
    Process:
    1. Find all .py files
    2. AST analysis of each file
    3. Extract entities and relations
    4. Resolve cross-file references
    5. Return integrated graph
  
  build_from_file(file_path: str) → CodeGraph
    Analyze single Python file

  _analyze_file(file_path)
    Parse Python AST and extract structure
  
  _resolve_references()
    Link unresolved cross-file references
"""

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
EXAMPLE 1: Manual CRG Construction
──────────────────────────────────

from src.phase1_causal_analysis import (
    CodeGraph, CodeEntity, CodeRelation, EntityType, RelationType,
    FailureEvidence, CRGBuilder, DataFlowWeighting
)

# 1. Build code graph
cg = CodeGraph()
file = CodeEntity(id="bug.py", name="bug.py", entity_type=EntityType.FILE, 
                  file_path="bug.py")
cg.add_entity(file)

method = CodeEntity(id="calc::add", name="add", entity_type=EntityType.FUNCTION,
                   file_path="bug.py", parent_id="calc")
cg.add_entity(method)

rel = CodeRelation(source_id="bug.py", target_id="calc::add",
                   relation_type=RelationType.CONTAINS)
cg.add_relation(rel)

# 2. Create failure evidence
failure = FailureEvidence(
    symptom_type="assertion_failure",
    symptom_location="test.py::test_add",
    symptom_message="assert 5 == 6",
    test_case_id="test_1",
    stack_trace=["test.py:25: test_add", "bug.py:8: add"]
)

# 3. Build CRG
builder = CRGBuilder(cg, DataFlowWeighting())
crg = builder.build([failure])

# 4. Collect paths
from src.phase1_causal_analysis import PathCollector
collector = PathCollector(crg)
paths = collector.collect_paths("test_add")

print(f"Found {len(paths)} candidate root causes")
for path in paths:
    print(f"  Path: {' ← '.join(path)}")


EXAMPLE 2: Automatic Analysis
──────────────────────────────

from src.phase1_causal_analysis import CodeGraphBuilder

builder = CodeGraphBuilder()
cg = builder.build_from_repository("/path/to/buggy/repo")

# Now use CG for CRG construction...
crg_builder = CRGBuilder(cg, DataFlowWeighting())
crg = crg_builder.build([failure])
"""

# ============================================================================
# ADVANCED TOPICS
# ============================================================================

"""
CUSTOM WEIGHTING STRATEGIES
───────────────────────────

The DataFlowWeighting is a simple strategy. You can implement custom ones:

from src.phase1_causal_analysis import EdgeWeightingStrategy

class CustomWeighting(EdgeWeightingStrategy):
    def compute_weight(self, source, target, relation_type, 
                      failures, code_graph):
        # Your logic here
        return weight  # in [0, 1]

# Then use:
crg_builder = CRGBuilder(cg, CustomWeighting())


FILTERING BY CONFIDENCE
──────────────────────

Prune low-confidence paths:

for path in paths:
    score = 1.0
    for i in range(len(path)-1):
        weight = crg.get_edge_weight(path[i], path[i+1])
        if weight:
            score *= weight
    
    if score > 0.3:  # Threshold
        high_confidence_paths.append(path)


MULTI-FAILURE SCENARIOS
──────────────────────

If test has multiple assertions:

failures = [
    FailureEvidence(..., symptom_message="assert a == 1"),
    FailureEvidence(..., symptom_message="assert b == 2"),
]

crg = builder.build(failures)

CRG will include paths explaining ALL failures,
helping pinpoint root cause affecting multiple symptoms.
"""

# ============================================================================
# NEXT PHASES
# ============================================================================

"""
After CRG Construction (Phase 1.1), continues with:

PHASE 1.2: MULTI-AGENT DEBATE
──────────────────────────────

Using CRG structure:
  • Each agent proposes a hypothesis from candidate paths
  • Agents debate evidence for/against each hypothesis
  • Debate refines confidence scores
  • Dynamic optimization strengthens winning paths

PHASE 1.3: DYNAMIC OPTIMIZATION
───────────────────────────────

Refine CRG based on debate:
  • Increase weights of supported paths
  • Decrease weights of refuted paths
  • Prune dominated candidates
  • Converge to most likely root cause

PHASE 1.4: RESULT SELECTION
──────────────────────────

Output final candidate locations:
  • Top-K root cause predictions
  • Supporting causal explanations
  • Confidence scores
  
  Pass to Phase 2 for repair generation
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nDocumentation loaded. See docstring for details.")
