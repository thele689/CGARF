import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.common.llm_interface import QwenLLMInterface
from src.srcd.consistency_distiller import ConsistencyDistiller


class BatchSRCDDistillationRunner:
    """Run paper section 3.2.3 on top of saved 3.2.2 reflection output."""

    def __init__(
        self,
        workspace_root: str,
        llm_client=None,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        embedding_cache_dir: Optional[str] = None,
        device: str = "cpu",
        embedding_backend_type: str = "auto",
    ):
        self.workspace_root = Path(workspace_root)
        self.distiller = ConsistencyDistiller(
            llm=llm_client,
            embedding_model_name=embedding_model,
            embedding_cache_dir=embedding_cache_dir,
            device=device,
            embedding_backend_type=embedding_backend_type,
        )

    def _load_reflection_payload(
        self,
        instance_id: str,
        reflection_json: Optional[str] = None,
    ) -> Optional[dict]:
        path = Path(reflection_json) if reflection_json else (
            self.workspace_root / "data" / "srcd_reflection" / f"{instance_id}_srcd_reflection.json"
        )
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_result(self, instance_id: str, payload: dict) -> str:
        out_dir = self.workspace_root / "data" / "srcd_distillation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{instance_id}_srcd_distillation.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return str(out_path)

    def process_instance(
        self,
        instance_id: str,
        reflection_json: Optional[str] = None,
        top_k_per_candidate: int = 5,
    ) -> Optional[str]:
        reflection_payload = self._load_reflection_payload(instance_id, reflection_json=reflection_json)
        if reflection_payload is None:
            logger.error(f"Missing SRCD reflection payload for {instance_id}")
            return None

        distilled = self.distiller.distill_reflection_payload(
            reflection_payload=reflection_payload,
            top_k_per_candidate=top_k_per_candidate,
        )
        payload = {
            "instance_id": instance_id,
            "reflection_source": reflection_json,
            **distilled,
        }
        saved_path = self._save_result(instance_id, payload)
        logger.success(f"SRCD 3.2.3 result saved to {saved_path}")
        return saved_path


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Run SRCD 3.2.3 consistency distillation.")
    parser.add_argument("--workspace-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--reflection-json", default=None)
    parser.add_argument("--top-k-per-candidate", type=int, default=5)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-cache-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "siliconflow", "local"],
        default="auto",
        help="Embedding backend for 3.2.3. auto uses SiliconFlow API when SILICONFLOW_API_KEY is set.",
    )
    args = parser.parse_args()

    runner = BatchSRCDDistillationRunner(
        workspace_root=args.workspace_root,
        llm_client=QwenLLMInterface(),
        embedding_model=args.embedding_model,
        embedding_cache_dir=args.embedding_cache_dir,
        device=args.device,
        embedding_backend_type=args.embedding_backend,
    )
    runner.process_instance(
        instance_id=args.instance_id,
        reflection_json=args.reflection_json,
        top_k_per_candidate=args.top_k_per_candidate,
    )
