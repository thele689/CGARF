"""LLM Interface Layer for CGARF"""

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum
import logging

from loguru import logger


class AgentType(Enum):
    """Types of agents in debate system"""
    SUPPORT = "support"
    OPPOSE = "oppose"
    JUDGE = "judge"


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    QWEN = "qwen"
    LOCAL = "local"


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.call_count = 0
        self.token_stats = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        self.call_history = []
        self.logger = logger

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, 
                max_tokens: int = 2000, **kwargs) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def generate_with_schema(self, prompt: str, output_schema: Dict[str, Any],
                           temperature: float = 0.7) -> Dict[str, Any]:
        """Generate text with JSON schema validation"""
        pass

    def compare_relative(self, objects: List[str], question: str) -> Dict[str, Any]:
        """LLM relative comparison of multiple objects"""
        
        prompt = f"""You are comparing multiple items and determining which one best answers the question.

Question: {question}

Items to compare:
"""
        for i, obj in enumerate(objects):
            prompt += f"\n{i+1}. {obj}\n"

        prompt += """
Please analyze each item and determine which one is better for the question.
Return your response in JSON format:
{
    "winner_idx": <index (0-based) of the winning item>,
    "confidence": <confidence score 0-1>,
    "reasoning": "<brief explanation>"
}"""

        response = self.generate_with_schema(prompt, {
            "winner_idx": int,
            "confidence": float,
            "reasoning": str
        })

        return response

    def generate_semantic_summary(self, code: str, lines_limit: int = 2) -> str:
        """Generate semantic summary for code"""
        
        prompt = f"""Analyze the following code and provide a brief semantic summary in {lines_limit} lines.
Code:
{code}

Summary (max {lines_limit} lines):"""

        summary = self.generate(prompt, temperature=0.5, max_tokens=500)
        return summary.strip()

    def evaluate_reflection(self, issue: str, code: str, path: str, 
                          patch: str, temperature: float = 0.7) -> Dict[str, Any]:
        """Self-reflection evaluation of a patch"""
        
        prompt = f"""Evaluate the following patch on three dimensions: semantic consistency, 
causal alignment, and minimal edit.

Issue Description:
{issue}

Target Code:
{code}

Representative Path (why this location is important):
{path}

Proposed Patch:
{patch}

Evaluate on three dimensions:
1. Semantic Consistency (sem): Does the patch preserve original semantics?
2. Causal Alignment (caus): Does the patch align with the causal path?
3. Minimal Edit (min): Is the modification scope minimal?

For each dimension, choose one of: fully_not, partially_not, neutral, partially_yes, fully_yes

Return JSON:
{{
    "semantic_consistency": {{"level": "<level>", "reason": "<reason>"}},
    "causal_alignment": {{"level": "<level>", "reason": "<reason>"}},
    "minimal_edit": {{"level": "<level>", "reason": "<reason>"}},
    "revision_suggestion": "<suggestion for next iteration>"
}}"""

        response = self.generate_with_schema(prompt, {
            "semantic_consistency": {"level": str, "reason": str},
            "causal_alignment": {"level": str, "reason": str},
            "minimal_edit": {"level": str, "reason": str},
            "revision_suggestion": str
        }, temperature=temperature)

        return response

    def extract_consensus_pattern(self, patches: List[str]) -> str:
        """Extract consensus repair pattern from multiple patches"""
        
        prompt = f"""Analyze the following patches and extract the consensus repair pattern.
What are the common changes, targets, and mechanisms across these patches?

Patches:
"""
        for i, patch in enumerate(patches):
            prompt += f"\n{i+1}.\n{patch}\n"

        prompt += """
Summarize the shared repair intent and mechanism:"""

        response = self.generate(prompt, temperature=0.5, max_tokens=800)
        return response.strip()

    def agent_debate(self, agent_type: AgentType, issue: str, 
                    path: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Run agent debate for path evaluation"""
        
        if agent_type == AgentType.SUPPORT:
            prompt = f"""You are a supporting agent evaluating a proposed root cause location.

Issue: {issue}

Proposed Causal Path:
{path}

Your task: Argue why this path plausibly explains the failure.
Focus on: failure entry point alignment, mechanism chain coherence, key nodes significance.

Return JSON:
{{
    "claim": "<main claim>",
    "evidence_refs": ["<node1>", "<edge2>", ...],
    "mechanism": "<detailed mechanism explanation>"
}}"""

        elif agent_type == AgentType.OPPOSE:
            prompt = f"""You are a skeptical agent evaluating a proposed root cause location.

Issue: {issue}

Proposed Causal Path:
{path}

Your task: Identify weaknesses in this path explanation.
Focus on: semantic gaps, weak connections, unrelated nodes, missing explanations.

Return JSON:
{{
    "counter_claim": "<main counter-claim>",
    "weak_links": ["<weak_edge1>", "<weak_node2>", ...],
    "noise_flags": ["<noise_type1>", "<noise_type2>", ...]
}}"""

        elif agent_type == AgentType.JUDGE:
            prompt = f"""You are a judge evaluating two arguments about a causal path.

Issue: {issue}

Proposed Path:
{path}

Supporting Argument: {context.get('support', '')}
Opposing Argument: {context.get('oppose', '')}

Your task: Decide which argument is more credible based on failure entry affinity, chain coherence,
and mechanism validity.

Return JSON:
{{
    "winner": "support|oppose",
    "key_reasons": ["<reason1>", "<reason2>", ...],
    "evidence_refs": ["<supporting_ref1>", ...]
}}"""

        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        response = self.generate_with_schema(prompt, {}, temperature=0.7)
        return response

    def generate_patch(self, issue: str, code: str, path: str, 
                      temperature: float = 0.7, round: int = 0) -> str:
        """Generate a patch for the code"""
        
        prompt = f"""Generate a patch to fix the following issue.

Issue Description:
{issue}

Target Code:
{code}

Representative Causal Path (why these locations are important):
{path}

Requirements:
1. Keep modifications minimal and localized
2. Preserve original semantics and functionality
3. Align with the causal path where issues originate
4. Use Search/Replace format

Output Format:
<<< SEARCH
<original code>
===
<fixed code>
>>> REPLACE

Patch (round {round}):"""

        patch = self.generate(prompt, temperature=temperature, max_tokens=2000)
        return patch.strip()

    def _call_with_retry(self, prompt: str, schema: Optional[Dict] = None,
                        max_retries: int = 3, **kwargs) -> str:
        """Call LLM with retry logic"""
        
        for attempt in range(max_retries):
            try:
                if schema:
                    result = self.generate_with_schema(prompt, schema, **kwargs)
                    return json.dumps(result)
                else:
                    return self.generate(prompt, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise

    def _validate_and_parse(self, response: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Validate and parse response"""
        
        try:
            # Try to parse as JSON
            data = json.loads(response)
            
            # Validate schema if provided
            if schema:
                self._validate_schema(data, schema)
            
            return data
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            raise ValueError(f"Could not extract valid JSON from response: {response}")

    def _validate_schema(self, data: Dict, schema: Dict) -> bool:
        """Validate data against schema"""
        
        for key, expected_type in schema.items():
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
            
            if not isinstance(data[key], expected_type):
                if isinstance(expected_type, dict):
                    # Nested schema
                    self._validate_schema(data[key], expected_type)
                else:
                    raise ValueError(
                        f"Invalid type for {key}: expected {expected_type}, "
                        f"got {type(data[key])}"
                    )
        
        return True

    def update_token_stats(self, prompt_tokens: int, completion_tokens: int):
        """Update token statistics"""
        
        self.token_stats['prompt_tokens'] += prompt_tokens
        self.token_stats['completion_tokens'] += completion_tokens
        self.token_stats['total_tokens'] += prompt_tokens + completion_tokens

    def get_stats(self) -> Dict[str, Any]:
        """Get call statistics"""
        
        return {
            'total_calls': self.call_count,
            'token_stats': self.token_stats,
            'average_tokens_per_call': (
                self.token_stats['total_tokens'] // self.call_count
                if self.call_count > 0 else 0
            )
        }

    def log_call(self, prompt: str, response: str, duration: float):
        """Log LLM call details"""
        
        self.call_history.append({
            'timestamp': time.time(),
            'duration': duration,
            'model': self.model_name,
            'prompt_length': len(prompt),
            'response_length': len(response)
        })


class OpenAILLMInterface(LLMInterface):
    """OpenAI or OpenAI-compatible chat completions API implementation."""

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        super().__init__(model_name)

        try:
            from openai import OpenAI

            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
            resolved_api_base = api_base or os.getenv("OPENAI_API_BASE")
            client_kwargs = {"api_key": resolved_api_key}
            if resolved_api_base:
                client_kwargs["base_url"] = resolved_api_base
            self.api_key = resolved_api_key
            self.api_base = resolved_api_base
            self.client = OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    def generate(self, prompt: str, temperature: float = 0.7,
                max_tokens: int = 2000, **kwargs) -> str:
        """Generate using OpenAI API"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            completion = response.choices[0].message.content
            
            # Update stats
            self.call_count += 1
            self.update_token_stats(
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            
            return completion
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise

    def generate_with_schema(self, prompt: str, output_schema: Dict[str, Any],
                           temperature: float = 0.7) -> Dict[str, Any]:
        """Generate with structured output"""
        
        response_text = self.generate(prompt, temperature=temperature)
        return self._validate_and_parse(response_text, output_schema)


class MockLLMInterface(LLMInterface):
    """Mock LLM for testing"""

    def __init__(self, model_name: str = "mock", responses: Optional[Dict] = None):
        super().__init__(model_name)
        self.responses = responses or {}
        self.call_log = []

    def generate(self, prompt: str, temperature: float = 0.7,
                max_tokens: int = 2000, **kwargs) -> str:
        """Return mock response"""
        
        self.call_count += 1
        self.call_log.append(prompt)
        
        # Return matching response or default
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response
        
        return "Mock response placeholder"

    def generate_with_schema(self, prompt: str, output_schema: Dict[str, Any],
                           temperature: float = 0.7) -> Dict[str, Any]:
        """Return mock structured response"""
        
        response_text = self.generate(prompt, temperature=temperature)
        return {"result": response_text}


class QwenLLMInterface(LLMInterface):
    """OpenAI-compatible Qwen interface for SiliconFlow/DashScope/OpenRouter endpoints."""

    @staticmethod
    def _mask_secret(secret: str) -> str:
        if not secret:
            return "<empty>"
        if len(secret) <= 10:
            return "*" * len(secret)
        return f"{secret[:5]}...{secret[-4:]}"

    def __init__(self, model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct", 
                 api_key: Optional[str] = None,
                 api_base: str = "https://api.siliconflow.cn/v1"):
        super().__init__(model_name)
        self.request_timeout_seconds = float(os.getenv("QWEN_REQUEST_TIMEOUT_SECONDS", "30"))
        self.max_retry_attempts = int(os.getenv("QWEN_MAX_RETRY_ATTEMPTS", "3"))
        self.min_request_interval_seconds = float(os.getenv("QWEN_MIN_REQUEST_INTERVAL_SECONDS", "0"))
        self.rate_limit_backoff_seconds = float(os.getenv("QWEN_RATE_LIMIT_BACKOFF_SECONDS", "20"))
        self._last_request_time = 0.0
        
        import requests
        if api_key:
            resolved_api_key = api_key
            api_key_source = "argument"
        elif os.getenv("SILICONFLOW_API_KEY"):
            resolved_api_key = os.getenv("SILICONFLOW_API_KEY", "")
            api_key_source = "env:SILICONFLOW_API_KEY"
        elif os.getenv("DASHSCOPE_API_KEY"):
            resolved_api_key = os.getenv("DASHSCOPE_API_KEY", "")
            api_key_source = "env:DASHSCOPE_API_KEY"
        elif os.getenv("OPENROUTER_API_KEY"):
            resolved_api_key = os.getenv("OPENROUTER_API_KEY", "")
            api_key_source = "env:OPENROUTER_API_KEY"
        elif os.getenv("QWEN_API_KEY"):
            resolved_api_key = os.getenv("QWEN_API_KEY", "")
            api_key_source = "env:QWEN_API_KEY"
        elif os.getenv("OPENAI_API_KEY"):
            resolved_api_key = os.getenv("OPENAI_API_KEY", "")
            api_key_source = "env:OPENAI_API_KEY"
        else:
            resolved_api_key = ""
            api_key_source = "unset"

        if os.getenv("SILICONFLOW_API_BASE"):
            resolved_api_base = os.getenv("SILICONFLOW_API_BASE", "")
            api_base_source = "env:SILICONFLOW_API_BASE"
        elif os.getenv("DASHSCOPE_API_BASE"):
            resolved_api_base = os.getenv("DASHSCOPE_API_BASE", "")
            api_base_source = "env:DASHSCOPE_API_BASE"
        elif os.getenv("OPENROUTER_API_BASE"):
            resolved_api_base = os.getenv("OPENROUTER_API_BASE", "")
            api_base_source = "env:OPENROUTER_API_BASE"
        elif os.getenv("QWEN_API_BASE"):
            resolved_api_base = os.getenv("QWEN_API_BASE", "")
            api_base_source = "env:QWEN_API_BASE"
        else:
            resolved_api_base = api_base
            api_base_source = "argument-default"

        self.api_key = resolved_api_key
        self.api_base = resolved_api_base
        self.requests = requests
        self.api_key_source = api_key_source
        self.api_base_source = api_base_source
        force_requests = os.getenv("QWEN_FORCE_REQUESTS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        self.logger.info(
            "Initialized QwenLLMInterface model={} base={} ({}) key={} ({})",
            self.model_name,
            self.api_base,
            self.api_base_source,
            self._mask_secret(self.api_key),
            self.api_key_source,
        )
        
        # Try to use OpenAI client if available for compatibility
        self.use_openai = False
        if force_requests:
            self.logger.info("QWEN_FORCE_REQUESTS is enabled; using direct requests for Qwen API")
        try:
            if force_requests:
                raise RuntimeError("direct requests forced by QWEN_FORCE_REQUESTS")
            from openai import OpenAI
            try:
                import httpx
                http_client = httpx.Client(trust_env=False, timeout=60.0)
            except Exception:
                http_client = None
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                http_client=http_client,
            )
            self.use_openai = True
            self.logger.info("Using OpenAI client for Qwen API")
        except Exception as e:
            self.logger.info("OpenAI client unavailable, using direct requests: {}", e)

    def _should_retry(self, error: Exception) -> bool:
        error_text = str(error).lower()
        transient_markers = [
            "timeout",
            "timed out",
            "rate limit",
            "429",
            "connection",
            "temporarily unavailable",
            "server error",
            "502",
            "503",
            "504",
        ]
        return any(marker in error_text for marker in transient_markers)

    def _is_rate_limited(self, error: Exception) -> bool:
        error_text = str(error).lower()
        return "429" in error_text or "rate limit" in error_text or "tpm limit" in error_text

    def _retry_delay_seconds(self, attempt_index: int, error: Optional[Exception] = None) -> float:
        if error is not None and self._is_rate_limited(error):
            return min(self.rate_limit_backoff_seconds * (2 ** attempt_index), 180.0)
        return min(2 ** attempt_index, 8.0)

    def _throttle_request(self) -> None:
        if self.min_request_interval_seconds <= 0:
            return
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_request_interval_seconds:
            time.sleep(self.min_request_interval_seconds - elapsed)

    def _mark_request_finished(self) -> None:
        self._last_request_time = time.time()

    def generate(self, prompt: str, temperature: float = 0.7,
                max_tokens: int = 2000, **kwargs) -> str:
        """Generate using Qwen API"""
        
        if self.use_openai:
            return self._generate_with_openai(prompt, temperature, max_tokens, **kwargs)
        else:
            return self._generate_with_requests(prompt, temperature, max_tokens, **kwargs)

    def _default_extra_body(self) -> Dict[str, Any]:
        if "dashscope.aliyuncs.com/compatible-mode" in (self.api_base or ""):
            return {"enable_thinking": True}
        if "openrouter.ai" in (self.api_base or ""):
            return {"reasoning": {"enabled": True}}
        return {}
    
    def _generate_with_openai(self, prompt: str, temperature: float = 0.7,
                              max_tokens: int = 2000, **kwargs) -> str:
        """Generate using OpenAI compatible client"""
        messages = kwargs.pop("messages", [{"role": "user", "content": prompt}])
        extra_body = kwargs.pop("extra_body", None)
        if extra_body is None:
            extra_body = self._default_extra_body()

        request_args = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": kwargs.pop("timeout", self.request_timeout_seconds),
        }
        if extra_body:
            request_args["extra_body"] = extra_body
        if kwargs:
            request_args.update(kwargs)

        for attempt in range(self.max_retry_attempts):
            try:
                self._throttle_request()
                response = self.client.chat.completions.create(**request_args)
                self._mark_request_finished()
                completion = response.choices[0].message.content

                self.call_count += 1
                if hasattr(response, 'usage'):
                    self.update_token_stats(
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens
                    )

                return completion
            except Exception as e:
                self._mark_request_finished()
                final_attempt = attempt == self.max_retry_attempts - 1
                self.logger.warning(
                    "Qwen OpenAI-client attempt {}/{} failed base={} model={} key={} source={}: {}",
                    attempt + 1,
                    self.max_retry_attempts,
                    self.api_base,
                    self.model_name,
                    self._mask_secret(self.api_key),
                    self.api_key_source,
                    e,
                )
                if final_attempt or not self._should_retry(e):
                    self.logger.error(
                        "Qwen API error via OpenAI client base={} key={} source={}: {}",
                        self.api_base,
                        self._mask_secret(self.api_key),
                        self.api_key_source,
                        e,
                    )
                    raise
                delay = self._retry_delay_seconds(attempt, e)
                self.logger.info("Retrying Qwen OpenAI-client call in {:.1f}s", delay)
                time.sleep(delay)
    
    def _generate_with_requests(self, prompt: str, temperature: float = 0.7,
                                max_tokens: int = 2000, **kwargs) -> str:
        """Generate using direct HTTP requests"""
        
        url = f"{self.api_base}/chat/completions"
        messages = kwargs.pop("messages", [{"role": "user", "content": prompt}])
        extra_body = kwargs.pop("extra_body", None)
        if extra_body is None:
            extra_body = self._default_extra_body()
        timeout = kwargs.pop("timeout", self.request_timeout_seconds)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            data.update(extra_body)
        if kwargs:
            data.update(kwargs)

        for attempt in range(self.max_retry_attempts):
            try:
                self._throttle_request()
                response = self.requests.post(url, json=data, headers=headers, timeout=timeout)
                self._mark_request_finished()
                response.raise_for_status()

                result = response.json()
                completion = result["choices"][0]["message"]["content"]

                self.call_count += 1
                if "usage" in result:
                    self.update_token_stats(
                        result["usage"].get("prompt_tokens", 0),
                        result["usage"].get("completion_tokens", 0)
                    )

                return completion
            except Exception as e:
                self._mark_request_finished()
                final_attempt = attempt == self.max_retry_attempts - 1
                self.logger.warning(
                    "Qwen requests-client attempt {}/{} failed base={} model={} key={} source={}: {}",
                    attempt + 1,
                    self.max_retry_attempts,
                    self.api_base,
                    self.model_name,
                    self._mask_secret(self.api_key),
                    self.api_key_source,
                    e,
                )
                if final_attempt or not self._should_retry(e):
                    self.logger.error("Qwen API error: {}", e)
                    raise
                delay = self._retry_delay_seconds(attempt, e)
                self.logger.info("Retrying Qwen requests-client call in {:.1f}s", delay)
                time.sleep(delay)

    def generate_with_schema(self, prompt: str, output_schema: Dict[str, Any],
                           temperature: float = 0.7) -> Dict[str, Any]:
        """Generate with structured output"""
        
        response_text = self.generate(prompt, temperature=temperature)
        return self._validate_and_parse(response_text, output_schema)


def create_llm_interface(provider: str, model_name: str, **kwargs) -> LLMInterface:
    """Factory function to create LLM interface"""

    if provider == "openai":
        return OpenAILLMInterface(model_name, **kwargs)
    elif provider in {"vllm", "openai-compatible"}:
        kwargs.setdefault("api_key", os.getenv("VLLM_API_KEY", "EMPTY"))
        kwargs.setdefault("api_base", os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"))
        return OpenAILLMInterface(model_name, **kwargs)
    elif provider == "qwen":
        return QwenLLMInterface(model_name, **kwargs)
    elif provider == "mock":
        return MockLLMInterface(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
