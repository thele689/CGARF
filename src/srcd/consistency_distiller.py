"""Consistency distillation for SRCD.

This module keeps the legacy clustering-based interface used by older tests,
and also provides a paper-aligned section 3.2.3 implementation operating on
the structured 3.2.2 reflection payload.
"""

import json
import os
import re
import ast
import time
import difflib
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from src.common.llm_interface import LLMInterface
from src.common.data_structures import PatchCandidate
from src.srcd.repair_generator import RepairCandidate
from src.srcd.reflection_scorer import ReflectionScore


class RepairPattern(Enum):
    """Recognized repair patterns"""
    NULL_CHECK = "null_check"
    EXCEPTION_HANDLER = "exception_handler"
    FALLBACK = "fallback"
    VALIDATION = "validation"
    GUARD_CLAUSE = "guard_clause"
    VARIABLE_ASSIGNMENT = "variable_assignment"
    TYPE_CONVERSION = "type_conversion"


@dataclass
class DistilledRepair:
    """Distilled repair with aggregated scores"""
    repair: RepairCandidate
    reflection_score: float
    consensus_score: float
    embedding_similarity: float
    distillation_score: float
    patterns: List[RepairPattern]
    confidence: float


class PatternExtractor:
    """Extracts repair patterns from code"""
    
    def __init__(self):
        """Initialize pattern extractor"""
        self.logger = logger
        self.pattern_signatures = self._define_patterns()
    
    def _define_patterns(self) -> Dict[RepairPattern, List[str]]:
        """Define pattern signatures"""
        
        return {
            RepairPattern.NULL_CHECK: [
                r'if\s+\w+\s+is\s+not\s+None',
                r'if\s+\w+\s+is\s+not\s+None\s+and',
                r'isinstance\(',
            ],
            RepairPattern.EXCEPTION_HANDLER: [
                r'try\s*:',
                r'except\s+\w+\s*:',
                r'except\s+Exception\s*:',
            ],
            RepairPattern.FALLBACK: [
                r'\w+\s+or\s+',
                r'if\s+\w+\s+is\s+None\s+else',
                r'\.get\(',
            ],
            RepairPattern.VALIDATION: [
                r'isinstance\(',
                r'len\(',
                r'validate',
            ],
            RepairPattern.GUARD_CLAUSE: [
                r'if\s+\w+\s*:',
                r'if\s+not\s+',
                r'return\s+',
            ],
            RepairPattern.VARIABLE_ASSIGNMENT: [
                r'\w+\s*=\s*',
                r'def\s+\w+',
            ],
            RepairPattern.TYPE_CONVERSION: [
                r'str\(',
                r'int\(',
                r'float\(',
                r'list\(',
                r'dict\(',
            ],
        }
    
    def extract_patterns(self, code: str) -> List[RepairPattern]:
        """
        Extract patterns from repair code
        
        Args:
            code: Repair code
        
        Returns:
            List of identified patterns
        """
        
        patterns = []
        
        for pattern_type, signatures in self.pattern_signatures.items():
            for signature in signatures:
                if re.search(signature, code):
                    patterns.append(pattern_type)
                    break  # Each pattern type counted once
        
        return patterns
    
    def extract_all_repairs(self, repairs: List[RepairCandidate]) -> Dict[str, List[RepairPattern]]:
        """Extract patterns for all repairs"""
        
        all_patterns = {}
        
        for repair in repairs:
            patterns = self.extract_patterns(repair.repaired_code)
            all_patterns[repair.id] = patterns
        
        return all_patterns


class EmbeddingClusterer:
    """Clusters repairs by semantic similarity"""
    
    def __init__(self):
        """Initialize embedding clusterer"""
        self.logger = logger
        
        # Try to import embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_embeddings = True
        except:
            self.logger.warning("Sentence transformers not available")
            self.use_embeddings = False
    
    def cluster_repairs(
        self,
        repairs: List[RepairCandidate],
        n_clusters: int = 5
    ) -> Dict[int, List[RepairCandidate]]:
        """
        Cluster repairs by semantic similarity
        
        Args:
            repairs: List of repairs
            n_clusters: Number of clusters
        
        Returns:
            Dict mapping cluster_id -> list of repairs
        """
        
        if not repairs:
            return {}
        
        # Limit clusters to number of repairs
        n_clusters = min(n_clusters, len(repairs))
        
        if self.use_embeddings:
            # Embed repairs
            embeddings = []
            for repair in repairs:
                try:
                    embedding = self.embedder.encode(repair.repaired_code, convert_to_tensor=False)
                    embeddings.append(embedding)
                except Exception as e:
                    self.logger.warning(f"Embedding failed: {e}")
                    embeddings.append(np.random.randn(384))  # Random fallback
            
            embeddings = np.array(embeddings)
            
            # K-means clustering
            try:
                from sklearn.cluster import KMeans
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
            
            except ImportError:
                # Fallback: simple distance-based clustering
                self.logger.warning("scikit-learn not available, using simple clustering")
                labels = self._simple_cluster(embeddings, n_clusters)
        
        else:
            # Fallback: simple clustering
            labels = self._simple_cluster_code(repairs, n_clusters)
        
        # Group by cluster
        clusters = {}
        for repair, label in zip(repairs, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(repair)
        
        self.logger.info(f"Clustered {len(repairs)} repairs into {len(clusters)} clusters")
        
        return clusters
    
    def _simple_cluster(self, embeddings: np.ndarray, n_clusters: int) -> List[int]:
        """Simple clustering without sklearn"""
        
        n_samples = len(embeddings)
        labels = np.zeros(n_samples, dtype=int)
        
        # Random initialization
        centers = embeddings[np.random.choice(n_samples, n_clusters, replace=False)]
        
        # Simple nearest neighbor assignment
        for i, embedding in enumerate(embeddings):
            distances = [np.linalg.norm(embedding - center) for center in centers]
            labels[i] = np.argmin(distances)
        
        return labels.tolist()
    
    def _simple_cluster_code(
        self,
        repairs: List[RepairCandidate],
        n_clusters: int
    ) -> List[int]:
        """Cluster repairs by code similarity (simple approach)"""
        
        labels = []
        
        for i, repair in enumerate(repairs):
            # Assign to cluster based on repair characteristics
            cluster_id = i % n_clusters
            labels.append(cluster_id)
        
        return labels
    
    def compute_cluster_centroid(
        self,
        cluster_repairs: List[RepairCandidate]
    ) -> Optional[np.ndarray]:
        """Compute cluster centroid (average embedding)"""
        
        if not self.use_embeddings or not cluster_repairs:
            return None
        
        embeddings = []
        
        for repair in cluster_repairs:
            try:
                embedding = self.embedder.encode(repair.repaired_code, convert_to_tensor=False)
                embeddings.append(embedding)
            except:
                continue
        
        if not embeddings:
            return None
        
        return np.mean(embeddings, axis=0)
    
    def similarity_to_cluster(
        self,
        repair: RepairCandidate,
        cluster_centroid: np.ndarray
    ) -> float:
        """Compute similarity between repair and cluster centroid"""
        
        if not self.use_embeddings or cluster_centroid is None:
            return 0.5
        
        try:
            repair_embedding = self.embedder.encode(repair.repaired_code, convert_to_tensor=False)
            
            # Cosine similarity
            sim = np.dot(repair_embedding, cluster_centroid) / (
                np.linalg.norm(repair_embedding) * np.linalg.norm(cluster_centroid)
            )
            
            # Scale to [0, 1]
            return (sim + 1) / 2
        
        except:
            return 0.5


@dataclass
class ConsensusPatternSummary:
    """Structured consensus repair pattern from appendix D.3."""

    candidate_id: str
    candidate_location: str
    shared_edit_intent: str
    shared_target_entities: List[str]
    shared_mechanism: str
    shared_constraints: List[str]
    source: str = "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_location": self.candidate_location,
            "shared_edit_intent": self.shared_edit_intent,
            "shared_target_entities": list(self.shared_target_entities),
            "shared_mechanism": self.shared_mechanism,
            "shared_constraints": list(self.shared_constraints),
            "source": self.source,
        }

    def to_text(self) -> str:
        targets = ", ".join(self.shared_target_entities) if self.shared_target_entities else "none"
        constraints = "; ".join(self.shared_constraints) if self.shared_constraints else "none"
        return (
            f"Shared edit intent: {self.shared_edit_intent}\n"
            f"Shared target entities: {targets}\n"
            f"Shared mechanism: {self.shared_mechanism}\n"
            f"Shared constraints: {constraints}"
        )


@dataclass
class SRCDDistilledPatch:
    """Paper-aligned distilled patch score for one candidate-round patch."""

    patch_id: str
    candidate_id: str
    candidate_location: str
    generated_round: int
    patch_content: str
    embedding_text: str
    reflection_score: float
    causality_score: float
    consistency_score: float
    embedding_similarity: float
    dispersion_penalty: float
    distillation_score: float
    is_no_op_patch: bool = False
    kept: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "candidate_id": self.candidate_id,
            "candidate_location": self.candidate_location,
            "generated_round": self.generated_round,
            "patch_content": self.patch_content,
            "embedding_text": self.embedding_text,
            "reflection_score": self.reflection_score,
            "causality_score": self.causality_score,
            "consistency_score": self.consistency_score,
            "embedding_similarity": self.embedding_similarity,
            "dispersion_penalty": self.dispersion_penalty,
            "distillation_score": self.distillation_score,
            "is_no_op_patch": self.is_no_op_patch,
            "kept": self.kept,
        }


class PaperConsensusExtractor:
    """LLM-backed consensus repair pattern extraction for one candidate patch set."""

    def __init__(self, llm: Optional[LLMInterface] = None):
        self.llm = llm
        self.logger = logger

    def extract(
        self,
        candidate_id: str,
        candidate_location: str,
        code_context: str,
        patch_items: List[Dict[str, Any]],
    ) -> ConsensusPatternSummary:
        if self.llm:
            prompt = self._build_prompt(candidate_location, code_context, patch_items)
            try:
                response_text = self.llm.generate(prompt, temperature=0.2, max_tokens=1000)
                payload = self._parse_consensus_json(response_text)
                return ConsensusPatternSummary(
                    candidate_id=candidate_id,
                    candidate_location=candidate_location,
                    shared_edit_intent=str(payload.get("shared_edit_intent", "")).strip(),
                    shared_target_entities=[
                        str(item).strip()
                        for item in payload.get("shared_target_entities", [])
                        if str(item).strip()
                    ],
                    shared_mechanism=str(payload.get("shared_mechanism", "")).strip(),
                    shared_constraints=[
                        str(item).strip()
                        for item in payload.get("shared_constraints", [])
                        if str(item).strip()
                    ],
                    source="llm_json",
                )
            except Exception as exc:
                self.logger.warning(f"Consensus extraction failed for {candidate_id}: {exc}")

        return self._heuristic_extract(candidate_id, candidate_location, code_context, patch_items)

    def _parse_consensus_json(self, response_text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in consensus response")
        payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ValueError("Consensus response is not a JSON object")
        return payload

    def _build_prompt(
        self,
        candidate_location: str,
        code_context: str,
        patch_items: List[Dict[str, Any]],
    ) -> str:
        patch_list = []
        for item in patch_items:
            patch_list.append(
                f"Patch {item['generated_round']} ({item['patch_id']}):\n{item['patch_content']}"
            )

        patch_text = "\n\n".join(patch_list)
        return f"""You are the consensus extraction agent in an automated program repair system.
Given a set of candidate patches produced for the same candidate location, summarize the repeated repair pattern they share.

Do not describe each patch separately. Instead, summarize their commonality in terms of:
1. what is mainly being edited,
2. how the edit is mainly performed,
3. what failure mechanism the edits are collectively trying to fix,
4. what local constraints the edits appear to preserve.

Return strict JSON only.

Candidate location:
{candidate_location}

Candidate code context:
```python
{code_context}
```

Candidate patch set:
{patch_text}

Output JSON:
{{
  "shared_edit_intent": "...",
  "shared_target_entities": ["..."],
  "shared_mechanism": "...",
  "shared_constraints": ["..."]
}}
""".strip()

    def _heuristic_extract(
        self,
        candidate_id: str,
        candidate_location: str,
        code_context: str,
        patch_items: List[Dict[str, Any]],
    ) -> ConsensusPatternSummary:
        target_entities: List[str] = []
        mechanisms: List[str] = []
        constraints = ["local edit", "keep interface unchanged"]

        for item in patch_items:
            search_block, replace_block = self._extract_patch_blocks(item["patch_content"])
            target_entities.extend(self._extract_identifiers(search_block + "\n" + replace_block))
            mechanisms.append(self._infer_mechanism(search_block, replace_block))

        unique_targets = []
        for target in target_entities:
            if target not in unique_targets:
                unique_targets.append(target)
            if len(unique_targets) >= 5:
                break

        mechanism = mechanisms[0] if mechanisms else "local logic adjustment"
        if any("condition" in item for item in mechanisms):
            mechanism = "adjust local condition or branch handling"
        elif any("return" in item for item in mechanisms):
            mechanism = "adjust local return or output computation"

        candidate_symbol = candidate_location.split("::")[-1]
        return ConsensusPatternSummary(
            candidate_id=candidate_id,
            candidate_location=candidate_location,
            shared_edit_intent=f"Modify local logic inside {candidate_symbol} to better align with the failure mechanism.",
            shared_target_entities=unique_targets or [candidate_symbol],
            shared_mechanism=mechanism,
            shared_constraints=constraints,
            source="heuristic",
        )

    def _extract_patch_blocks(self, patch: str) -> Tuple[str, str]:
        match = re.search(r'<<<\s*SEARCH\n(.*?)\n===\n(.*?)\n>>>\s*REPLACE', patch, re.DOTALL)
        if not match:
            return "", ""
        return match.group(1).strip(), match.group(2).strip()

    def _extract_identifiers(self, text: str) -> List[str]:
        identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)
        filtered = []
        for item in identifiers:
            if item in {"if", "else", "return", "def", "try", "except", "True", "False"}:
                continue
            if item not in filtered:
                filtered.append(item)
        return filtered

    def _infer_mechanism(self, search_block: str, replace_block: str) -> str:
        if "if " in replace_block and "if " not in search_block:
            return "add or strengthen a local condition"
        if "isinstance(" in replace_block:
            return "tighten local type-sensitive branching"
        if "return " in replace_block and replace_block != search_block:
            return "adjust local return or output computation"
        return "modify local computation logic"


class TextEmbeddingBackend:
    """Sentence embedding backend for paper section 3.2.3."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        encoder: Any = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.encoder = encoder
        self.logger = logger
        self._model = None
        self._tokenizer = None
        self._mode = None
        self.load_error: Optional[str] = None
        self.fallback_model_name: Optional[str] = None

    def _load_model(self):
        if self.encoder is not None or self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        kwargs: Dict[str, Any] = {}
        if self.cache_dir:
            kwargs["cache_folder"] = self.cache_dir
        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
                **kwargs,
            )
            self._mode = "sentence_transformers"
        except TypeError:
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    **kwargs,
                )
                self._mode = "sentence_transformers"
            except Exception as exc:
                self.load_error = str(exc)
                self._load_hf_model_or_fallback()
        except Exception as exc:
            self.load_error = str(exc)
            self._load_hf_model_or_fallback()

    def _load_hf_model_or_fallback(self):
        try:
            self._load_hf_model()
        except Exception as exc:
            previous_error = self.load_error
            self.load_error = str(exc)
            fallback_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.logger.warning(
                "Failed to load embedding model {} on {}. Falling back to {}. "
                "Initial error: {}. HF error: {}",
                self.model_name,
                self.device,
                fallback_name,
                previous_error,
                self.load_error,
            )
            from sentence_transformers import SentenceTransformer

            kwargs: Dict[str, Any] = {}
            if self.cache_dir:
                kwargs["cache_folder"] = self.cache_dir
            self._model = SentenceTransformer(
                fallback_name,
                device=self.device,
                **kwargs,
            )
            self._mode = "sentence_transformers_fallback"
            self.fallback_model_name = fallback_name

    def _load_hf_model(self):
        from transformers import AutoModel, AutoTokenizer

        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        self._model = AutoModel.from_pretrained(self.model_name, **kwargs)
        self._model.to(self.device)
        self._model.eval()
        self._mode = "hf_auto"

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=float)
        if self.encoder is not None:
            embeddings = self.encoder.encode(texts)
        else:
            self._load_model()
            if self._mode == "sentence_transformers":
                embeddings = self._model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
            else:
                embeddings = self._encode_with_hf_model(texts)
        return np.asarray(embeddings, dtype=float)

    def _encode_with_hf_model(self, texts: List[str]) -> np.ndarray:
        import torch

        if hasattr(self._model, "encode"):
            try:
                embeddings = self._model.encode(texts)
                return np.asarray(embeddings, dtype=float)
            except Exception:
                pass

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**encoded)

        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return pooled.cpu().numpy()

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output.cpu().numpy()

        if isinstance(outputs, torch.Tensor):
            return outputs.cpu().numpy()

        raise ValueError("Unsupported embedding model output format")

    def cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        cosine = float(np.dot(left, right) / (left_norm * right_norm))
        return max(0.0, min(1.0, cosine))

    @property
    def effective_model_name(self) -> str:
        return self.fallback_model_name or self.model_name

    @property
    def mode(self) -> Optional[str]:
        return self._mode


class SiliconFlowEmbeddingBackend:
    """SiliconFlow embedding API backend for paper section 3.2.3."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        api_key: Optional[str] = None,
        api_base: str = "https://api.siliconflow.cn/v1",
        timeout_seconds: float = 30.0,
        max_retry_attempts: int = 3,
        session: Any = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY", "")
        self.api_base = (api_base or os.getenv("SILICONFLOW_API_BASE") or "https://api.siliconflow.cn/v1").rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retry_attempts = max_retry_attempts
        self.session = session
        self.load_error: Optional[str] = None
        self.logger = logger
        self._mode = "siliconflow_api"

    def _client(self):
        if self.session is not None:
            return self.session
        import requests

        self.session = requests.Session()
        self.session.trust_env = False
        return self.session

    def _should_retry(self, error: Exception) -> bool:
        text = str(error).lower()
        return any(
            marker in text
            for marker in ["timeout", "429", "rate", "temporarily", "502", "503", "504", "connection"]
        )

    def _post_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("SILICONFLOW_API_KEY is required for SiliconFlow embedding API")

        url = f"{self.api_base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "input": texts if len(texts) != 1 else texts[0],
        }

        for attempt in range(self.max_retry_attempts):
            try:
                response = self._client().post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                final_attempt = attempt == self.max_retry_attempts - 1
                self.load_error = str(exc)
                self.logger.warning(
                    "SiliconFlow embedding attempt {}/{} failed model={} base={}: {}",
                    attempt + 1,
                    self.max_retry_attempts,
                    self.model_name,
                    self.api_base,
                    exc,
                )
                if final_attempt or not self._should_retry(exc):
                    raise
                time.sleep(min(2 ** attempt, 8))

        raise RuntimeError("unreachable SiliconFlow embedding retry state")

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=float)

        unique_texts: List[str] = []
        text_to_unique_index: Dict[str, int] = {}
        reconstruction_indices: List[int] = []
        for text in texts:
            if text not in text_to_unique_index:
                text_to_unique_index[text] = len(unique_texts)
                unique_texts.append(text)
            reconstruction_indices.append(text_to_unique_index[text])

        payload = self._post_embeddings(unique_texts)
        data = payload.get("data", [])
        if not isinstance(data, list) or not data:
            raise ValueError(f"Invalid SiliconFlow embedding response: {payload}")

        data = sorted(data, key=lambda item: item.get("index", 0))
        embeddings = [item.get("embedding") for item in data]
        if any(not isinstance(item, list) for item in embeddings):
            raise ValueError(f"Missing embedding vectors in SiliconFlow response: {payload}")
        unique_embeddings = np.asarray(embeddings, dtype=float)
        return np.asarray([unique_embeddings[index] for index in reconstruction_indices], dtype=float)

    def cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        cosine = float(np.dot(left, right) / (left_norm * right_norm))
        return max(0.0, min(1.0, cosine))

    @property
    def effective_model_name(self) -> str:
        return self.model_name

    @property
    def mode(self) -> str:
        return self._mode


class ConsistencyDistiller:
    """Main consistency distillation orchestrator"""
    
    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        embedding_cache_dir: Optional[str] = None,
        device: str = "cpu",
        embedding_backend: Optional[TextEmbeddingBackend] = None,
        embedding_backend_type: str = "auto",
    ):
        """
        Initialize consistency distiller
        
        Args:
            llm: LLM interface
        """
        self.llm = llm
        self.logger = logger
        
        self.pattern_extractor: Optional[PatternExtractor] = None
        self.clusterer: Optional[EmbeddingClusterer] = None
        self.consensus_extractor = PaperConsensusExtractor(llm)
        self.embedding_backend = embedding_backend or self._create_embedding_backend(
            embedding_model_name=embedding_model_name,
            embedding_cache_dir=embedding_cache_dir,
            device=device,
            backend_type=embedding_backend_type,
        )
        
        # Weights from paper (Equation 10)
        self.alpha = 0.8  # Consensus weight
        self.beta = 0.5   # Consistency penalty

    def _create_embedding_backend(
        self,
        embedding_model_name: str,
        embedding_cache_dir: Optional[str],
        device: str,
        backend_type: str,
    ):
        normalized = (backend_type or "auto").lower()
        use_siliconflow = normalized == "siliconflow" or (
            normalized == "auto"
            and bool(os.getenv("SILICONFLOW_API_KEY"))
            and embedding_model_name.startswith("Qwen/")
        )
        if use_siliconflow:
            return SiliconFlowEmbeddingBackend(
                model_name=embedding_model_name,
                api_key=os.getenv("SILICONFLOW_API_KEY", ""),
                api_base=os.getenv("SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1"),
            )
        return TextEmbeddingBackend(
            model_name=embedding_model_name,
            cache_dir=embedding_cache_dir,
            device=device,
        )
    
    def distill_repairs(
        self,
        repairs: List[RepairCandidate],
        reflection_scores: Dict[str, ReflectionScore],
        n_clusters: int = 5
    ) -> List[DistilledRepair]:
        """
        Distill repairs using consensus patterns
        
        Formula: Dist(p, n) = Ref(p, n) + α·Cons(p) - β·(1 - cos_sim(p, cluster))
        
        Args:
            repairs: List of repair candidates
            reflection_scores: Dict mapping repair_id -> ReflectionScore
            n_clusters: Number of clusters for pattern consensus
        
        Returns:
            List of distilled repairs sorted by distillation score
        """
        
        self.logger.info(f"Distilling {len(repairs)} repairs")
        if self.pattern_extractor is None:
            self.pattern_extractor = PatternExtractor()
        if self.clusterer is None:
            self.clusterer = EmbeddingClusterer()
        
        # Step 1: Extract patterns
        all_patterns = self.pattern_extractor.extract_all_repairs(repairs)
        
        # Step 2: Cluster repairs
        clusters = self.clusterer.cluster_repairs(repairs, n_clusters)
        
        # Step 3: Extract consensus per cluster
        consensus_patterns = {}
        cluster_centroids = {}
        
        for cluster_id, cluster_repairs in clusters.items():
            consensus = self._extract_consensus(cluster_repairs, all_patterns)
            consensus_patterns[cluster_id] = consensus
            
            centroid = self.clusterer.compute_cluster_centroid(cluster_repairs)
            cluster_centroids[cluster_id] = centroid
        
        # Step 4: Distillation score
        distilled = []
        
        for repair in repairs:
            # Get reflection score
            if repair.id not in reflection_scores:
                self.logger.warning(f"No reflection score for {repair.id}")
                continue
            
            ref_score = reflection_scores[repair.id].combined_reflection
            
            # Find which cluster this repair belongs to
            cluster_id = self._find_repair_cluster(repair, clusters)
            
            # Consensus score
            cons_score = self._consensus_agreement(
                repair, consensus_patterns[cluster_id]
            )
            
            # Embedding similarity to cluster
            embedding_sim = self.clusterer.similarity_to_cluster(
                repair, cluster_centroids[cluster_id]
            )
            
            # Distillation score (Equation 10)
            dist_score = (
                ref_score +
                self.alpha * cons_score -
                self.beta * (1 - embedding_sim)
            )
            
            # Confidence combines reflection and consistency
            confidence = 0.6 * reflection_scores[repair.id].confidence + 0.4 * cons_score
            
            distilled_repair = DistilledRepair(
                repair=repair,
                reflection_score=ref_score,
                consensus_score=cons_score,
                embedding_similarity=embedding_sim,
                distillation_score=dist_score,
                patterns=all_patterns[repair.id],
                confidence=confidence
            )
            
            distilled.append(distilled_repair)
        
        # Step 5: Sort by distillation score
        distilled.sort(key=lambda x: x.distillation_score, reverse=True)
        
        self.logger.info(f"Distilled to {len(distilled)} ranked repairs")
        
        return distilled
    
    def _extract_consensus(
        self,
        cluster_repairs: List[RepairCandidate],
        all_patterns: Dict[str, List[RepairPattern]]
    ) -> List[RepairPattern]:
        """
        Extract consensus patterns from cluster
        
        Args:
            cluster_repairs: Repairs in cluster
            all_patterns: All extracted patterns
        
        Returns:
            List of consensus patterns (>50% agreement)
        """
        
        if not cluster_repairs:
            return []
        
        # Count pattern occurrences
        pattern_counts = {}
        
        for repair in cluster_repairs:
            patterns = all_patterns.get(repair.id, [])
            for pattern in patterns:
                if pattern not in pattern_counts:
                    pattern_counts[pattern] = 0
                pattern_counts[pattern] += 1
        
        # Select patterns with >50% agreement
        threshold = len(cluster_repairs) * 0.5
        consensus = [
            pattern for pattern, count in pattern_counts.items()
            if count > threshold
        ]
        
        return consensus
    
    def _consensus_agreement(
        self,
        repair: RepairCandidate,
        consensus_patterns: List[RepairPattern]
    ) -> float:
        """
        Compute agreement with consensus patterns
        
        Args:
            repair: Repair candidate
            consensus_patterns: Consensus from cluster
        
        Returns:
            Agreement score [0, 1]
        """
        
        if not consensus_patterns:
            return 0.5  # No consensus
        
        # Extract this repair's patterns
        patterns = self.pattern_extractor.extract_patterns(repair.repaired_code)
        
        # Compute overlap
        overlap = len(set(patterns) & set(consensus_patterns))
        total = len(set(patterns) | set(consensus_patterns))
        
        if total == 0:
            return 0.5
        
        return overlap / total
    
    def _find_repair_cluster(
        self,
        repair: RepairCandidate,
        clusters: Dict[int, List[RepairCandidate]]
    ) -> int:
        """Find which cluster a repair belongs to"""
        
        for cluster_id, cluster_repairs in clusters.items():
            if any(r.id == repair.id for r in cluster_repairs):
                return cluster_id
        
        # Fallback
        return 0
    
    def get_top_repairs(
        self,
        distilled_repairs: List[DistilledRepair],
        top_k: int = 10
    ) -> List[DistilledRepair]:
        """Get top-K distilled repairs"""
        
        return distilled_repairs[:top_k]

    def distill_reflection_payload(
        self,
        reflection_payload: Dict[str, Any],
        top_k_per_candidate: int = 5,
    ) -> Dict[str, Any]:
        """Paper-aligned section 3.2.3 distillation on structured 3.2.2 output."""

        candidate_results: Dict[str, Any] = {}
        candidate_runs = reflection_payload.get("candidate_runs", {})

        for candidate_id, run in candidate_runs.items():
            rounds = run.get("rounds", [])
            if not rounds:
                continue

            patch_items = []
            patch_embedding_texts = []
            for round_info in rounds:
                reflection = round_info.get("reflection", {})
                patch = round_info.get("patch", {})
                patch_content = str(patch.get("patch_content", ""))
                embedding_text = self._build_patch_embedding_text(patch_content)
                patch_embedding_texts.append(embedding_text)
                patch_items.append(
                    {
                        "patch_id": str(patch.get("patch_id", "")),
                        "generated_round": int(patch.get("generated_round", round_info.get("round", 0) or 0)),
                        "patch_content": patch_content,
                        "embedding_text": embedding_text,
                        "is_no_op_patch": self._is_search_replace_no_op(patch_content),
                        "reflection_score": float(reflection.get("combined_reflection", 0.0)),
                        "causality_score": self._extract_reflection_causality_score(reflection),
                        "reflection": reflection,
                    }
                )

            consensus_pattern = self.consensus_extractor.extract(
                candidate_id=candidate_id,
                candidate_location=str(run.get("candidate_location", candidate_id)),
                code_context=self._recover_code_context(rounds),
                patch_items=patch_items,
            )

            patch_embeddings = self.embedding_backend.encode_texts(patch_embedding_texts)
            consensus_embedding = self.embedding_backend.encode_texts([consensus_pattern.to_text()])[0]
            centroid = np.mean(patch_embeddings, axis=0)
            dispersion_penalty = float(
                np.mean(
                    [
                        1.0 - self.embedding_backend.cosine_similarity(centroid, patch_embedding)
                        for patch_embedding in patch_embeddings
                    ]
                )
            ) if len(patch_embeddings) else 0.0

            ranked: List[SRCDDistilledPatch] = []
            for item, patch_embedding in zip(patch_items, patch_embeddings):
                consistency_score = self.embedding_backend.cosine_similarity(
                    patch_embedding,
                    consensus_embedding,
                )
                embedding_similarity = self.embedding_backend.cosine_similarity(
                    patch_embedding,
                    centroid,
                )
                distillation_score = (
                    item["reflection_score"] +
                    self.alpha * consistency_score -
                    self.beta * dispersion_penalty
                )
                ranked.append(
                    SRCDDistilledPatch(
                        patch_id=item["patch_id"],
                        candidate_id=candidate_id,
                        candidate_location=str(run.get("candidate_location", candidate_id)),
                        generated_round=item["generated_round"],
                        patch_content=item["patch_content"],
                        embedding_text=item["embedding_text"],
                        reflection_score=item["reflection_score"],
                        causality_score=item["causality_score"],
                        consistency_score=consistency_score,
                        embedding_similarity=embedding_similarity,
                        dispersion_penalty=dispersion_penalty,
                        distillation_score=distillation_score,
                        is_no_op_patch=bool(item["is_no_op_patch"]),
                    )
                )

            ranked.sort(key=lambda item: item.distillation_score, reverse=True)
            for patch in ranked[:top_k_per_candidate]:
                patch.kept = True

            candidate_results[candidate_id] = {
                "candidate_id": candidate_id,
                "candidate_location": str(run.get("candidate_location", candidate_id)),
                "allocated_samples": int(run.get("allocated_samples", len(ranked))),
                "unique_patch_count": len(set(item["patch_content"] for item in patch_items)),
                "unique_embedding_text_count": len(set(item["embedding_text"] for item in patch_items)),
                "no_op_patch_count": sum(1 for item in patch_items if item["is_no_op_patch"]),
                "consensus_pattern": consensus_pattern.to_dict(),
                "dispersion_penalty": dispersion_penalty,
                "ranked_patches": [patch.to_dict() for patch in ranked],
                "kept_patch_ids": [patch.patch_id for patch in ranked[:top_k_per_candidate]],
            }

        return {
            "llm_model": getattr(self.llm, "model_name", None),
            "llm_call_count": getattr(self.llm, "call_count", None),
            "embedding_model": getattr(self.embedding_backend, "model_name", None),
            "effective_embedding_model": getattr(self.embedding_backend, "effective_model_name", None),
            "embedding_mode": getattr(self.embedding_backend, "mode", None),
            "embedding_load_error": getattr(self.embedding_backend, "load_error", None),
            "alpha": self.alpha,
            "beta": self.beta,
            "top_k_per_candidate": top_k_per_candidate,
            "candidate_results": candidate_results,
        }

    def _recover_code_context(self, rounds: List[Dict[str, Any]]) -> str:
        """Recover per-candidate code context from patch SEARCH blocks when no direct field is present."""

        for round_info in rounds:
            patch = str(round_info.get("patch", {}).get("patch_content", ""))
            match = re.search(r'<<<\s*SEARCH\n(.*?)\n===\n', patch, re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""

    def _build_patch_embedding_text(self, patch_content: str) -> str:
        """Represent a patch by its actual edit rather than repeated SEARCH context."""

        search_block, replace_block = self._extract_search_replace_blocks(patch_content)
        if not search_block and not replace_block:
            return self._truncate_for_embedding(patch_content)

        removed_lines, added_lines = self._changed_lines(search_block, replace_block)
        if not removed_lines and not added_lines:
            return "No textual edit; SEARCH and REPLACE blocks are identical."

        return self._truncate_for_embedding(
            "Removed lines:\n"
            + "\n".join(removed_lines[:40])
            + "\n\nAdded lines:\n"
            + "\n".join(added_lines[:40])
        )

    def _extract_search_replace_blocks(self, patch_content: str) -> Tuple[str, str]:
        match = re.search(r'<<<\s*SEARCH\n(.*?)\n===\n(.*?)\n>>>\s*REPLACE', patch_content, re.DOTALL)
        if not match:
            return "", ""
        return match.group(1).strip(), match.group(2).strip()

    def _is_search_replace_no_op(self, patch_content: str) -> bool:
        search_block, replace_block = self._extract_search_replace_blocks(patch_content)
        if not search_block and not replace_block:
            return False
        return search_block.strip() == replace_block.strip()

    def _extract_reflection_causality_score(self, reflection: Dict[str, Any]) -> float:
        if "causal_score" in reflection:
            return max(0.0, min(1.0, float(reflection.get("causal_score") or 0.0)))
        causal_alignment = reflection.get("causal_alignment", {}) or {}
        if isinstance(causal_alignment, dict) and "score" in causal_alignment:
            return max(0.0, min(1.0, float(causal_alignment.get("score") or 0.0)))
        return 0.0

    def _changed_lines(self, before: str, after: str) -> Tuple[List[str], List[str]]:
        diff = difflib.ndiff(before.splitlines(), after.splitlines())
        removed: List[str] = []
        added: List[str] = []
        for line in diff:
            if line.startswith("- "):
                stripped = line[2:].strip()
                if stripped:
                    removed.append(stripped)
            elif line.startswith("+ "):
                stripped = line[2:].strip()
                if stripped:
                    added.append(stripped)
        return removed, added

    def _truncate_for_embedding(self, text: str, max_chars: int = 3000) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]"


class RepairAggregator:
    """Aggregates repair scores and statistics"""
    
    def __init__(self):
        """Initialize repair aggregator"""
        self.logger = logger
    
    def aggregate_scores(
        self,
        reflection: float,
        consensus: float,
        embedding_sim: float,
        alpha: float = 0.8,
        beta: float = 0.5
    ) -> float:
        """
        Aggregate scores using distillation formula
        
        Args:
            reflection: Reflection score [0, 1]
            consensus: Consensus agreement [0, 1]
            embedding_sim: Embedding similarity [0, 1]
            alpha: Consensus weight
            beta: Consistency penalty weight
        
        Returns:
            Aggregated distillation score
        """
        
        return (
            reflection +
            alpha * consensus -
            beta * (1 - embedding_sim)
        )
    
    def get_repair_statistics(
        self,
        distilled_repairs: List[DistilledRepair]
    ) -> Dict:
        """Get statistics about distilled repairs"""
        
        if not distilled_repairs:
            return {}
        
        scores = [r.distillation_score for r in distilled_repairs]
        
        return {
            'count': len(distilled_repairs),
            'avg_score': np.mean(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'std_score': np.std(scores),
            'median_score': np.median(scores),
        }
