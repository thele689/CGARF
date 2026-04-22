"""Batch entry point for paper section 3.3 TSPF.

This runner consumes the SRCD 3.2.3 distillation payload and applies the
two-stage patch filtering framework:

1. functional filtering with regression/reproduction evidence when available,
2. causality-matching plus group-similarity ranking.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from src.tspf.patch_filter import TwoStagePatchFilter
from src.tspf.agentless_adapted_tests import extract_tspf_evidence


class BatchTSPFRunner:
    """Run TSPF for one distilled SRCD payload."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root

    def process_instance(
        self,
        instance_id: str,
        distillation_json: Path,
        test_evidence_json: Optional[Path] = None,
        mu: float = 0.6,
        max_patches: int = 5,
        require_test_evidence: bool = True,
    ) -> Path:
        distillation_payload = json.loads(distillation_json.read_text())
        test_evidence: Optional[Dict[str, Dict[str, Any]]] = None
        if test_evidence_json:
            test_evidence = extract_tspf_evidence(json.loads(test_evidence_json.read_text()))

        tspf = TwoStagePatchFilter(
            mu=mu,
            require_test_evidence=require_test_evidence,
        )
        result = tspf.filter_distillation_payload(
            distillation_payload,
            test_evidence=test_evidence,
            max_patches=max_patches,
        )
        result.update(
            {
                "instance_id": instance_id,
                "source_distillation_json": str(distillation_json),
                "source_test_evidence_json": str(test_evidence_json) if test_evidence_json else None,
            }
        )

        output_dir = self.workspace_root / "data" / "tspf"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{instance_id}_tspf.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        logger.success("TSPF result saved to {}", output_path)
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Run paper-aligned TSPF section 3.3.")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--distillation-json", type=Path, required=True)
    parser.add_argument("--test-evidence-json", type=Path, default=None)
    parser.add_argument("--mu", type=float, default=0.6)
    parser.add_argument("--max-patches", type=int, default=5)
    parser.add_argument(
        "--allow-missing-test-evidence",
        action="store_true",
        help="Debug/compatibility mode. Paper-aligned default requires both regression and reproduction evidence.",
    )
    args = parser.parse_args()

    runner = BatchTSPFRunner(args.workspace_root)
    runner.process_instance(
        instance_id=args.instance_id,
        distillation_json=args.distillation_json,
        test_evidence_json=args.test_evidence_json,
        mu=args.mu,
        max_patches=args.max_patches,
        require_test_evidence=not args.allow_missing_test_evidence,
    )


if __name__ == "__main__":
    main()
