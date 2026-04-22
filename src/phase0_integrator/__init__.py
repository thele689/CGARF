"""
Phase 0: Unified Fault Localization + SWE-Bench Integration
============================================================

This module integrates multiple fault localization methods (OrcaLoca, Agentless, 
CoSIL) with SWE-Bench instances to create enhanced issue contexts for CGARF 
pipeline input.

Supported Methods:
  - orcaloca: OrcaLoca fault localization
  - agentless: Agentless fault localization
  - cosil: CoSIL fault localization

Main Classes:
  - UnifiedFaultLocalizationLoader: Unified loader for all methods
  - OrcaLocaDataLoader: Alias for backward compatibility
  - BugLocation: Represents a single bug location candidate
  - LocalizationOutput: Parsed localization output
  - SWEBenchInstance: SWE-Bench instance metadata
  - EnhancedIssueContext: Combined context from localization + SWE-Bench
"""

from .fault_localization_loader import (
    UnifiedFaultLocalizationLoader,
    BugLocation,
    LocalizationOutput,
    SWEBenchInstance,
    EnhancedIssueContext,
)

__all__ = [
    'UnifiedFaultLocalizationLoader',
    'BugLocation',
    'LocalizationOutput',
    'SWEBenchInstance',
    'EnhancedIssueContext',
]
