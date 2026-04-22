"""Structured self-reflection for SRCD section 3.2.2."""

import json
import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass, field
from loguru import logger

from src.common.llm_interface import LLMInterface
from src.common.data_structures import IssueContext, PathEvidence, PatchCandidate
from src.srcd.repair_generator import RepairCandidate, SRCDCandidateInput


LEVEL_TO_SCORE = {
    "fully_not": 0.0,
    "partially_not": 0.25,
    "neutral": 0.5,
    "partially_yes": 0.75,
    "fully_yes": 1.0,
}


@dataclass
class ReflectionDimension:
    level: str
    score: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReflectionScore:
    """Reflection score components"""
    repair_id: str
    semantic_score: float
    causal_score: float
    minimality_score: float
    combined_reflection: float
    confidence: float
    semantic_consistency: Optional[ReflectionDimension] = None
    causal_alignment: Optional[ReflectionDimension] = None
    minimal_edit: Optional[ReflectionDimension] = None
    revision_suggestion: str = ""
    suggested_temperature: Optional[float] = None
    source: str = "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "repair_id": self.repair_id,
            "semantic_score": self.semantic_score,
            "causal_score": self.causal_score,
            "minimality_score": self.minimality_score,
            "combined_reflection": self.combined_reflection,
            "confidence": self.confidence,
            "revision_suggestion": self.revision_suggestion,
            "suggested_temperature": self.suggested_temperature,
            "source": self.source,
        }
        if self.semantic_consistency is not None:
            payload["semantic_consistency"] = self.semantic_consistency.to_dict()
        if self.causal_alignment is not None:
            payload["causal_alignment"] = self.causal_alignment.to_dict()
        if self.minimal_edit is not None:
            payload["minimal_edit"] = self.minimal_edit.to_dict()
        return payload


class SemanticSimilarityEvaluator:
    """Evaluates semantic similarity between repair and issue"""
    
    def __init__(self, llm: Optional[LLMInterface] = None):
        """
        Initialize semantic evaluator
        
        Args:
            llm: LLM interface for semantic analysis
        """
        self.llm = llm
        self.logger = logger
        
        # Try to import embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_embeddings = True
        except:
            self.logger.warning("Sentence transformers not available, using LLM for similarity")
            self.use_embeddings = False
    
    def evaluate(self, repair_code: str, issue_description: str) -> float:
        """
        Evaluate semantic similarity
        
        Args:
            repair_code: Repair code
            issue_description: Issue description
        
        Returns:
            Similarity score [0, 1]
        """
        
        if self.use_embeddings:
            # Use embedding-based similarity
            try:
                repair_embedding = self.embedder.encode(repair_code, convert_to_tensor=False)
                issue_embedding = self.embedder.encode(issue_description, convert_to_tensor=False)
                
                # Cosine similarity
                similarity = np.dot(repair_embedding, issue_embedding) / (
                    np.linalg.norm(repair_embedding) * np.linalg.norm(issue_embedding)
                )
                
                # Clamp to [0, 1]
                return max(0.0, min(1.0, (similarity + 1) / 2))
            
            except Exception as e:
                self.logger.error(f"Embedding similarity failed: {e}")
                return 0.5
        
        elif self.llm:
            # Use LLM-based similarity
            prompt = f"""Rate how well this repair code addresses the issue (0-100):

Issue: {issue_description}

Repair: {repair_code[:200]}

Return only a number 0-100."""
            
            try:
                response = self.llm.generate(prompt)
                score = float(re.search(r'\d+', response).group()) / 100
                return max(0.0, min(1.0, score))
            except:
                return 0.5
        
        else:
            # Fallback: keyword matching
            keywords = set(issue_description.lower().split())
            repair_words = set(repair_code.lower().split())
            
            if len(keywords | repair_words) == 0:
                return 0.5
            
            overlap = len(keywords & repair_words)
            jaccard = overlap / len(keywords | repair_words)
            
            return jaccard


class CausalRelevanceEvaluator:
    """Evaluates causal relevance to bug root cause"""
    
    def __init__(self, llm: Optional[LLMInterface] = None):
        """
        Initialize causal evaluator
        
        Args:
            llm: LLM interface for impact analysis
        """
        self.llm = llm
        self.logger = logger
    
    def evaluate(
        self,
        repair_code: str,
        issue_location: str,
        crg_path: Optional[PathEvidence] = None,
        issue_type: Optional[str] = None
    ) -> float:
        """
        Evaluate causal relevance
        
        Args:
            repair_code: Repair code
            issue_location: Bug location
            crg_path: Causal relevance graph path
            issue_type: Type of error (NullPointerException, ValueError, etc.)
        
        Returns:
            Causal relevance score [0, 1]
        """
        
        causal_score = 0.0
        
        # Check 1: Modifies variables on path
        if crg_path:
            path_variables = self._extract_variables_from_path(crg_path)
            if self._modifies_path_variables(repair_code, path_variables):
                causal_score += 0.4
                self.logger.debug(f"Repair modifies path variables: +0.4")
        
        # Check 2: Adds guard/check at issue location
        if self._adds_guard_at_location(repair_code, issue_location):
            causal_score += 0.35
            self.logger.debug(f"Repair adds guard at location: +0.35")
        
        # Check 3: Handles error type
        if issue_type:
            if self._handles_error_type(repair_code, issue_type):
                causal_score += 0.25
                self.logger.debug(f"Repair handles {issue_type}: +0.25")
        
        # Use LLM for detailed causal analysis if available
        if self.llm:
            try:
                llm_score = self._llm_causal_analysis(
                    repair_code, issue_location, issue_type
                )
                # Combine with heuristic score
                causal_score = 0.5 * causal_score + 0.5 * llm_score
            except Exception as e:
                self.logger.warning(f"LLM causal analysis failed: {e}")
        
        return min(causal_score, 1.0)
    
    def _extract_variables_from_path(self, crg_path: PathEvidence) -> set:
        """Extract variable names from CRG path"""
        
        variables = set()
        
        # Variables from path nodes - extract entity_id from CRGNode objects
        for node in crg_path.nodes:
            if hasattr(node, 'entity_id'):
                variables.add(node.entity_id)
            elif isinstance(node, str):
                variables.add(node)
        
        # Extract from node names (if they're function/variable names)
        for node in crg_path.nodes:
            node_name = node.entity_id if hasattr(node, 'entity_id') else str(node)
            # Simple variable extraction
            if '_' in node_name:
                parts = node_name.split('_')
                variables.update(parts)
        
        return variables
    
    def _modifies_path_variables(self, repair_code: str, path_variables: set) -> bool:
        """Check if repair modifies variables on path"""
        
        if not path_variables:
            return False
        
        # Look for assignments to path variables
        for var in path_variables:
            # Pattern: var = something
            if re.search(rf'\b{var}\s*=', repair_code):
                return True
            
            # Pattern: var.method() or var[index]
            if re.search(rf'\b{var}[\.\[]', repair_code):
                return True
        
        return False
    
    def _adds_guard_at_location(self, repair_code: str, location: str) -> bool:
        """Check if repair adds guard/check at issue location"""
        
        guard_patterns = [
            r'if\s+\w+\s+is\s+not\s+None',  # if x is not None
            r'if\s+\w+:',                     # if x:
            r'isinstance\(',                  # isinstance check
            r'try:',                          # try-except
            r'assert\s+',                     # assertion
        ]
        
        for pattern in guard_patterns:
            if re.search(pattern, repair_code):
                return True
        
        return False
    
    def _handles_error_type(self, repair_code: str, error_type: str) -> bool:
        """Check if repair handles specific error type"""
        
        # Extract base error type
        base_error = error_type.split(':')[0] if ':' in error_type else error_type
        
        # Python exception patterns
        error_patterns = {
            'NullPointerException': [r'is\s+not\s+None', r'is\s+None'],
            'AttributeError': [r'hasattr\(', r'getattr\('],
            'ValueError': [r'isinstance\(', r'validate'],
            'TypeError': [r'isinstance\(', r'type\('],
            'IndexError': [r'len\(', r'\[.*\]'],
            'KeyError': [r'\.get\(', r'in\s+\w+'],
        }
        
        patterns = error_patterns.get(base_error, [])
        
        for pattern in patterns:
            if re.search(pattern, repair_code):
                return True
        
        return False
    
    def _llm_causal_analysis(
        self,
        repair_code: str,
        issue_location: str,
        issue_type: Optional[str] = None
    ) -> float:
        """Use LLM for detailed causal analysis"""
        
        prompt = f"""Analyze if this repair code causally addresses the bug root cause (0-100):

Bug Location: {issue_location}
Error Type: {issue_type or 'Unknown'}

Repair Code:
{repair_code[:300]}

Does this repair:
1. Modify critical variables? (yes/no)
2. Add guards at the failure point? (yes/no)
3. Handle the error type? (yes/no)

Return a confidence score 0-100."""
        
        try:
            response = self.llm.generate(prompt)
            score = float(re.search(r'\d+', response).group()) / 100
            return max(0.0, min(1.0, score))
        except:
            return 0.5


class MinimalityEvaluator:
    """Evaluates minimality of repairs (prefer small changes)"""
    
    def __init__(self):
        """Initialize minimality evaluator"""
        self.logger = logger
    
    def evaluate(self, repair_code: str, original_code: str) -> float:
        """
        Evaluate repair minimality
        
        Args:
            repair_code: Repaired code
            original_code: Original code
        
        Returns:
            Minimality score [0, 1] (higher = smaller change)
        """
        
        # Line-based distance
        original_lines = original_code.split('\n')
        repair_lines = repair_code.split('\n')
        
        lines_changed = abs(len(repair_lines) - len(original_lines))
        
        # Character-based distance
        chars_changed = self._edit_distance(original_code, repair_code)
        
        # Exponential penalty for large changes
        penalty = np.exp(-0.1 * lines_changed)
        
        # Also consider character changes
        char_ratio = chars_changed / max(len(original_code), 1)
        char_penalty = np.exp(-0.05 * char_ratio)
        
        # Combined minimality score
        minimality = 0.6 * penalty + 0.4 * char_penalty
        
        return max(0.0, min(1.0, minimality))
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Compute edit distance (Levenshtein) between two strings"""
        
        if len(s1) > 1000 or len(s2) > 1000:
            # Approximate for long strings
            common = len(set(s1.split()) & set(s2.split()))
            total = len(set(s1.split()) | set(s2.split()))
            return total - common if total > 0 else 0
        
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]


class ReflectionScorer:
    """Main reflection scoring orchestrator"""
    
    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        weights: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
        reflection_threshold: float = 0.75,
        temperature_step: float = 0.1,
    ):
        """
        Initialize reflection scorer
        
        Args:
            llm: LLM interface for semantic analysis
        """
        self.llm = llm
        self.logger = logger
        
        self.semantic_evaluator = SemanticSimilarityEvaluator(llm)
        self.causal_evaluator = CausalRelevanceEvaluator(llm)
        self.minimality_evaluator = MinimalityEvaluator()
        self.weight_semantic, self.weight_causal, self.weight_minimality = self._normalize_weights(weights)
        self.reflection_threshold = reflection_threshold
        self.temperature_step = temperature_step

    def score_patch_candidate(
        self,
        patch: PatchCandidate,
        candidate_input: SRCDCandidateInput,
        issue_text: str,
        current_temperature: float = 0.2,
    ) -> ReflectionScore:
        """Paper 3.2.2 entry point for one structured patch reflection."""

        prompt = self._build_reflection_prompt(
            issue_text=issue_text,
            candidate_input=candidate_input,
            patch_text=patch.patch_content,
        )
        payload = self._call_reflection_llm(prompt)
        if payload is None:
            return self._heuristic_structured_reflection(
                patch=patch,
                candidate_input=candidate_input,
                issue_text=issue_text,
                current_temperature=current_temperature,
            )

        semantic = self._dimension_from_payload(payload, "semantic_consistency")
        causal = self._dimension_from_payload(payload, "causal_alignment")
        minimal = self._dimension_from_payload(payload, "minimal_edit")
        combined = (
            self.weight_semantic * semantic.score
            + self.weight_causal * causal.score
            + self.weight_minimality * minimal.score
        )
        confidence = (semantic.score + causal.score + minimal.score) / 3
        suggested_temperature = self.adjust_sampling_temperature(current_temperature, combined)
        return ReflectionScore(
            repair_id=patch.patch_id,
            semantic_score=semantic.score,
            causal_score=causal.score,
            minimality_score=minimal.score,
            combined_reflection=combined,
            confidence=confidence,
            semantic_consistency=semantic,
            causal_alignment=causal,
            minimal_edit=minimal,
            revision_suggestion=str(payload.get("revision_suggestion", "")).strip(),
            suggested_temperature=suggested_temperature,
            source="llm_structured",
        )

    def score_patch_bundle(
        self,
        issue_text: str,
        candidate_inputs: List[SRCDCandidateInput],
        patches: List[PatchCandidate],
        current_temperature: float = 0.2,
    ) -> Dict[str, ReflectionScore]:
        """Reflect over one SRCD patch bundle produced by section 3.2.1."""

        inputs_by_id = {item.candidate_id: item for item in candidate_inputs}
        scores: Dict[str, ReflectionScore] = {}
        for patch in patches:
            candidate_input = inputs_by_id.get(patch.location)
            if candidate_input is None:
                self.logger.warning(f"Missing candidate input for patch reflection: {patch.location}")
                continue
            scores[patch.patch_id] = self.score_patch_candidate(
                patch=patch,
                candidate_input=candidate_input,
                issue_text=issue_text,
                current_temperature=current_temperature,
            )
        return scores

    def adjust_sampling_temperature(
        self,
        current_temperature: float,
        combined_reflection: float,
    ) -> float:
        """Lower temperature for high-quality patches, raise it otherwise."""

        if combined_reflection >= self.reflection_threshold:
            return max(0.0, current_temperature - self.temperature_step)
        return min(1.0, current_temperature + self.temperature_step)

    def _normalize_weights(self, weights: Tuple[float, float, float]) -> Tuple[float, float, float]:
        total = sum(weights)
        if total <= 0:
            return (1 / 3, 1 / 3, 1 / 3)
        return tuple(weight / total for weight in weights)
    
    def score_repair(
        self,
        repair: RepairCandidate,
        original_code: str,
        issue_context: IssueContext,
        crg_path: Optional[PathEvidence] = None
    ) -> ReflectionScore:
        """
        Score a repair using reflection formula
        
        Formula: Ref(p, n) = 0.2×Sem + 0.4×Caus + 0.2×Min
        
        Args:
            repair: Repair candidate
            original_code: Original code before repair
            issue_context: Issue description
            crg_path: CRG path (optional for causal analysis)
        
        Returns:
            ReflectionScore with component and combined scores
        """
        
        self.logger.debug(f"Scoring repair {repair.id}")
        
        # Component scores
        sem_score = self.semantic_evaluator.evaluate(
            repair.repaired_code, issue_context.description
        )
        
        # Extract error type from issue
        error_type = self._extract_error_type(issue_context.description)
        
        caus_score = self.causal_evaluator.evaluate(
            repair.repaired_code,
            repair.id,
            crg_path,
            error_type
        )
        
        min_score = self.minimality_evaluator.evaluate(
            repair.repaired_code, original_code
        )
        
        # Combined reflection score (Equation 9)
        reflection = (
            self.weight_semantic * sem_score +
            self.weight_causal * caus_score +
            self.weight_minimality * min_score
        )
        
        # Confidence score
        confidence = min(
            repair.confidence,  # Original repair confidence
            (sem_score + caus_score + min_score) / 3  # Average scoring confidence
        )

        legacy_candidate = SRCDCandidateInput(
            candidate_id=repair.id,
            candidate_location=repair.id,
            file_path="",
            entity_type="unknown",
            code_context=original_code,
            representative_path_id="",
            representative_path_summary="",
        )
        semantic_dimension = self._dimension_from_score(
            sem_score,
            self._semantic_reason(sem_score),
        )
        causal_dimension = self._dimension_from_score(
            caus_score,
            self._causal_reason(caus_score, legacy_candidate),
        )
        minimal_dimension = self._dimension_from_score(
            min_score,
            self._minimal_reason(min_score),
        )
        
        result = ReflectionScore(
            repair_id=repair.id,
            semantic_score=sem_score,
            causal_score=caus_score,
            minimality_score=min_score,
            combined_reflection=reflection,
            confidence=confidence,
            semantic_consistency=semantic_dimension,
            causal_alignment=causal_dimension,
            minimal_edit=minimal_dimension,
            revision_suggestion=self._revision_suggestion(
                semantic_dimension,
                causal_dimension,
                minimal_dimension,
            ),
            suggested_temperature=self.adjust_sampling_temperature(0.2, reflection),
            source="legacy_heuristic",
        )
        
        self.logger.debug(
            f"Scores: Sem={sem_score:.2f} Caus={caus_score:.2f} "
            f"Min={min_score:.2f} → Ref={reflection:.2f}"
        )
        
        return result

    def _build_reflection_prompt(
        self,
        issue_text: str,
        candidate_input: SRCDCandidateInput,
        patch_text: str,
    ) -> str:
        """Appendix D.2 structured reflection prompt."""
        search_block, replace_block = self._extract_patch_blocks(patch_text)

        return f"""You are the structured reflection agent in an automated program repair system.
Given an issue description, a candidate location, the local code context, a representative causal-path summary, and the current Search/Replace patch, evaluate the proposed edit itself.

Do not invent facts not grounded in the input.
Judge the three dimensions independently:

1. Semantic consistency
Check whether the patch preserves behavior unrelated to the target defect.
Focus on function responsibility, input/output behavior, return-value semantics, interface behavior, state updates, exception handling, and unrelated control flow.

2. Causal alignment
Check whether the patch is aligned with the representative causal path and inferred failure mechanism.
Focus on whether the edit touches key nodes, key conditions, or key relations on the path, and whether it explains why the failure symptom would be mitigated.

3. Minimal edit
Check whether the patch stays within the smallest necessary scope.
Focus on whether it avoids broad rewrites, unnecessary refactoring, renaming, or unrelated branches.

Use exactly one label per dimension:
fully_not / partially_not / neutral / partially_yes / fully_yes

Scoring rubric:
- Use fully_not only when the patch clearly violates that dimension.
- Use partially_not when there are concrete concerns but the violation is not absolute.
- Use neutral when evidence is mixed or insufficient.
- Use partially_yes when the patch is mostly reasonable but still imperfect.
- Use fully_yes only when the patch strongly satisfies the dimension.
- A patch can be weak on causal alignment but still moderate on semantic consistency or minimal edit. Do not force all three dimensions to the same label unless the evidence clearly supports that.
- Even if this candidate location may not be globally ideal, keep the judgment focused on the patch proposed at this candidate location, and make the revision suggestion actionable at the same location.

Issue description:
{issue_text}

Candidate location:
{candidate_input.candidate_location}

Candidate code context:
```python
{candidate_input.code_context}
```

Representative path summary:
{candidate_input.representative_path_summary}

Patch SEARCH block:
```python
{search_block}
```

Patch REPLACE block:
```python
{replace_block}
```

Return strict JSON only. No markdown fences. No extra explanation.

Output JSON:
{{
  "semantic_consistency": {{
    "level": "fully_not|partially_not|neutral|partially_yes|fully_yes",
    "reason": "..."
  }},
  "causal_alignment": {{
    "level": "fully_not|partially_not|neutral|partially_yes|fully_yes",
    "reason": "..."
  }},
  "minimal_edit": {{
    "level": "fully_not|partially_not|neutral|partially_yes|fully_yes",
    "reason": "..."
  }},
  "revision_suggestion": "One concrete next-step improvement that must stay at the same candidate location."
}}
""".strip()

    def _call_reflection_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        if not self.llm:
            return None
        try:
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=900)
        except Exception as exc:
            self.logger.warning(f"Structured reflection LLM call failed: {exc}")
            return None

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            return None
        try:
            payload = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None

        required_keys = {
            "semantic_consistency",
            "causal_alignment",
            "minimal_edit",
            "revision_suggestion",
        }
        if not required_keys.issubset(payload.keys()):
            return None
        return payload

    def _dimension_from_payload(self, payload: Dict[str, Any], key: str) -> ReflectionDimension:
        value = payload.get(key, {}) or {}
        level = str(value.get("level", "neutral")).strip()
        if level not in LEVEL_TO_SCORE:
            level = "neutral"
        return ReflectionDimension(
            level=level,
            score=LEVEL_TO_SCORE[level],
            reason=str(value.get("reason", "")).strip(),
        )

    def _heuristic_structured_reflection(
        self,
        patch: PatchCandidate,
        candidate_input: SRCDCandidateInput,
        issue_text: str,
        current_temperature: float,
    ) -> ReflectionScore:
        """
        Deterministic fallback for section 3.2.2 when no structured LLM output
        is available.
        """

        search_block, replace_block = self._extract_patch_blocks(patch.patch_content)
        semantic_score = self._heuristic_semantic_consistency(search_block, replace_block)
        causal_score = self._heuristic_causal_alignment(candidate_input, replace_block, issue_text)
        minimal_score = self.minimality_evaluator.evaluate(replace_block, search_block)

        semantic = self._dimension_from_score(
            semantic_score,
            self._semantic_reason(semantic_score),
        )
        causal = self._dimension_from_score(
            causal_score,
            self._causal_reason(causal_score, candidate_input),
        )
        minimal = self._dimension_from_score(
            minimal_score,
            self._minimal_reason(minimal_score),
        )
        combined = (
            self.weight_semantic * semantic.score
            + self.weight_causal * causal.score
            + self.weight_minimality * minimal.score
        )
        confidence = (semantic.score + causal.score + minimal.score) / 3
        return ReflectionScore(
            repair_id=patch.patch_id,
            semantic_score=semantic.score,
            causal_score=causal.score,
            minimality_score=minimal.score,
            combined_reflection=combined,
            confidence=confidence,
            semantic_consistency=semantic,
            causal_alignment=causal,
            minimal_edit=minimal,
            revision_suggestion=self._revision_suggestion(semantic, causal, minimal),
            suggested_temperature=self.adjust_sampling_temperature(current_temperature, combined),
            source="heuristic_structured",
        )

    def _extract_patch_blocks(self, patch_text: str) -> Tuple[str, str]:
        match = re.search(
            r'<<<\s*SEARCH\n(.*?)\n===\n(.*?)\n>>>\s*REPLACE',
            patch_text,
            re.DOTALL,
        )
        if not match:
            return ("", patch_text)
        return (match.group(1).strip(), match.group(2).strip())

    def _heuristic_semantic_consistency(self, original: str, revised: str) -> float:
        if not original or not revised:
            return 0.25
        signature_original = original.splitlines()[0].strip() if original.splitlines() else ""
        signature_revised = revised.splitlines()[0].strip() if revised.splitlines() else ""
        signature_same = float(signature_original == signature_revised)
        minimality = self.minimality_evaluator.evaluate(revised, original)
        return max(0.0, min(1.0, 0.6 * signature_same + 0.4 * minimality))

    def _heuristic_causal_alignment(
        self,
        candidate_input: SRCDCandidateInput,
        revised: str,
        issue_text: str,
    ) -> float:
        path_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", candidate_input.representative_path_summary))
        revised_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", revised))
        issue_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", issue_text))

        overlap_path = len(path_tokens & revised_tokens) / max(len(path_tokens), 1)
        overlap_issue = len(issue_tokens & revised_tokens) / max(len(issue_tokens), 1)
        return max(0.0, min(1.0, 0.7 * overlap_path + 0.3 * overlap_issue))

    def _dimension_from_score(self, score: float, reason: str) -> ReflectionDimension:
        if score <= 0.125:
            level = "fully_not"
        elif score <= 0.375:
            level = "partially_not"
        elif score <= 0.625:
            level = "neutral"
        elif score <= 0.875:
            level = "partially_yes"
        else:
            level = "fully_yes"
        return ReflectionDimension(level=level, score=LEVEL_TO_SCORE[level], reason=reason)

    def _semantic_reason(self, score: float) -> str:
        if score >= 0.75:
            return "Patch largely preserves the original function signature and local semantics."
        if score >= 0.5:
            return "Patch is moderately aligned but still changes some local behavior."
        return "Patch likely changes semantics beyond the bug-focused scope."

    def _causal_reason(self, score: float, candidate_input: SRCDCandidateInput) -> str:
        if score >= 0.75:
            return "Patch text overlaps with representative-path concepts and stays near the inferred causal mechanism."
        if score >= 0.5:
            return "Patch is partially connected to the representative path but the causal explanation is not strong."
        return (
            "Patch shows weak textual alignment with the representative path and may not target "
            f"the main mechanism around {candidate_input.candidate_location}."
        )

    def _minimal_reason(self, score: float) -> str:
        if score >= 0.75:
            return "Patch remains local and introduces only limited edits."
        if score >= 0.5:
            return "Patch size is moderate and may still include avoidable changes."
        return "Patch modifies too much relative to the original snippet."

    def _revision_suggestion(
        self,
        semantic: ReflectionDimension,
        causal: ReflectionDimension,
        minimal: ReflectionDimension,
    ) -> str:
        dimensions = [
            ("semantic_consistency", semantic.score),
            ("causal_alignment", causal.score),
            ("minimal_edit", minimal.score),
        ]
        weakest = min(dimensions, key=lambda item: item[1])[0]
        if weakest == "semantic_consistency":
            return "Keep the fix more local and avoid changing behavior unrelated to the failing symptom."
        if weakest == "causal_alignment":
            return "Align the patch more directly with the representative causal path and failure mechanism."
        return "Reduce edit scope and avoid unnecessary restructuring."
    
    def score_repairs(
        self,
        repairs: List[RepairCandidate],
        original_code: str,
        issue_context: IssueContext,
        crg_path: Optional[PathEvidence] = None
    ) -> Dict[str, ReflectionScore]:
        """
        Score multiple repairs
        
        Args:
            repairs: List of repair candidates
            original_code: Original code
            issue_context: Issue context
            crg_path: CRG path for causal analysis
        
        Returns:
            Dict mapping repair_id -> ReflectionScore
        """
        
        scores = {}
        
        for repair in repairs:
            score = self.score_repair(
                repair, original_code, issue_context, crg_path
            )
            scores[repair.id] = score
        
        # Sort by reflection score
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1].combined_reflection,
            reverse=True
        )
        
        self.logger.info(f"Scored {len(repairs)} repairs")
        self.logger.debug(f"Top 3: {[s[0] for s in sorted_scores[:3]]}")
        
        return dict(sorted_scores)
    
    def _extract_error_type(self, issue_description: str) -> str:
        """Extract error type from issue description"""
        
        patterns = [
            r'(NullPointerException|AttributeError)',
            r'(ValueError|TypeError)',
            r'(IndexError)',
            r'(KeyError)',
            r'(FileNotFoundError)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, issue_description)
            if match:
                return match.group(1)
        
        return 'Exception'
