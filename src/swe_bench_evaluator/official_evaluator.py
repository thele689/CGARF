"""
Phase 4: Official SWE-Bench Evaluation (按官方标准)
================================================

本模块处理补丁评估并输出官方SWE-Bench格式的预测和评估结果。

官方标准：
  - 预测格式：{"instance_id": "...", "model_name_or_path": "...", "model_patch": "..."}
  - 评估方式：docker-based (官方run_evaluation.py)
  - 指标：PA (Patch Application Rate), SR (Success Rate)

支持两种评估模式：
  1. 轻量级评估：直接用pytest （快速原型）
  2. 官方Docker评估：调用官方harness （最终评估）
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from loguru import logger


@dataclass
class Prediction:
    """官方SWE-Bench预测格式"""
    instance_id: str
    model_name_or_path: str
    model_patch: str


@dataclass
class PatchResult:
    """单个补丁的评估结果"""
    patch_id: str
    patch_content: str
    
    # 应用结果
    applied: bool = False
    application_error: str = ""
    
    # 测试结果
    fail_to_pass_results: Dict[str, bool] = field(default_factory=dict)
    pass_to_pass_results: Dict[str, bool] = field(default_factory=dict)
    
    # 指标
    fail_to_pass_rate: float = 0.0
    pass_to_pass_rate: float = 0.0
    
    # 汇总
    resolved: bool = False
    time_taken: float = 0.0
    error_message: str = ""


@dataclass
class EvaluationMetrics:
    """评估指标汇总"""
    instance_id: str
    model_name_or_path: str = "CGARF"
    
    total_patches: int = 0
    applied_patches: int = 0
    resolved_patches: int = 0
    
    # 官方标准指标
    patch_application_rate: float = 0.0      # PA = applied / total
    success_rate: float = 0.0                 # SR = resolved / applied
    
    # 详细信息
    patch_results: List[PatchResult] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "total_patches": self.total_patches,
            "applied_patches": self.applied_patches,
            "resolved_patches": self.resolved_patches,
            "patch_application_rate": f"{self.patch_application_rate:.1%}",
            "success_rate": f"{self.success_rate:.1%}",
            "timestamp": datetime.now().isoformat(),
        }


class OfficialSWEBenchEvaluator:
    """
    按官方标准评估补丁
    """
    
    def __init__(
        self,
        model_name: str = "CGARF",
        lightweight_mode: bool = True,
        timeout: int = 60
    ):
        """
        初始化评估器
        
        Args:
            model_name: 模型名称 (用于预测输出)
            lightweight_mode: True=快速pytest, False=需要官方harness
            timeout: 测试超时 (秒)
        """
        self.model_name = model_name
        self.lightweight_mode = lightweight_mode
        self.timeout = timeout
    
    def _clone_repo(
        self,
        repo_url: str,
        base_commit: str,
        work_dir: str
    ) -> bool:
        """克隆仓库到指定commit"""
        try:
            subprocess.run(
                ['git', 'clone', repo_url, work_dir],
                capture_output=True,
                timeout=300,
                check=True
            )
            
            subprocess.run(
                ['git', 'checkout', base_commit],
                cwd=work_dir,
                capture_output=True,
                timeout=60,
                check=True
            )
            
            logger.info(f"✓ Repository cloned and checked out to {base_commit}")
            return True
        
        except Exception as e:
            logger.warning(f"Failed to clone repo: {e}")
            return False
    
    def _apply_patch(self, patch_content: str, work_dir: str) -> bool:
        """应用补丁"""
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.patch',
                dir=work_dir,
                delete=False
            ) as f:
                f.write(patch_content)
                patch_file = f.name
            
            try:
                subprocess.run(
                    ['git', 'apply', patch_file],
                    cwd=work_dir,
                    capture_output=True,
                    timeout=30,
                    check=True
                )
                logger.info(f"✓ Patch applied successfully")
                return True
            finally:
                os.unlink(patch_file)
        
        except Exception as e:
            logger.warning(f"Failed to apply patch: {e}")
            return False
    
    def _run_tests(
        self,
        test_specs: Dict[str, List[str]],
        work_dir: str
    ) -> Dict[str, Dict[str, bool]]:
        """
        运行测试 (轻量级模式)
        
        Args:
            test_specs: {"FAIL_TO_PASS": [...], "PASS_TO_PASS": [...]}
            work_dir: 仓库目录
        
        Returns:
            {"FAIL_TO_PASS": {test: bool}, "PASS_TO_PASS": {test: bool}}
        """
        results = {}
        
        for test_type, test_paths in test_specs.items():
            results[test_type] = {}
            
            for test_path in test_paths:
                full_path = os.path.join(work_dir, test_path)
                
                if not os.path.exists(full_path):
                    logger.warning(f"Test not found: {test_path}")
                    results[test_type][test_path] = False
                    continue
                
                try:
                    result = subprocess.run(
                        ['pytest', full_path, '-xvs'],
                        cwd=work_dir,
                        capture_output=True,
                        timeout=self.timeout,
                        text=True
                    )
                    
                    passed = result.returncode == 0
                    results[test_type][test_path] = passed
                    
                    if passed:
                        logger.info(f"  ✓ {test_type}: {test_path}")
                    else:
                        logger.warning(f"  ✗ {test_type}: {test_path}")
                
                except subprocess.TimeoutExpired:
                    logger.warning(f"  ⏱ Timeout: {test_path}")
                    results[test_type][test_path] = False
                except Exception as e:
                    logger.warning(f"  ✗ Error: {test_path} - {e}")
                    results[test_type][test_path] = False
        
        return results
    
    def evaluate_patch_lightweight(
        self,
        patch_id: str,
        patch_content: str,
        instance_info: Dict[str, Any]
    ) -> PatchResult:
        """
        轻量级补丁评估 (不需要Docker)
        
        用于快速原型和验证。最终评估需要用官方Docker。
        """
        result = PatchResult(
            patch_id=patch_id,
            patch_content=patch_content
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. 克隆
            repo_url = instance_info.get('repo_url')
            base_commit = instance_info.get('base_commit')
            
            if not repo_url:
                result.application_error = "Unknown repository"
                return result
            
            work_dir = os.path.join(tmpdir, 'repo')
            if not self._clone_repo(repo_url, base_commit, work_dir):
                result.application_error = "Failed to clone repository"
                return result
            
            # 2. 应用补丁
            if not self._apply_patch(patch_content, work_dir):
                result.application_error = "Failed to apply patch"
                return result
            
            result.applied = True
            
            # 3. 运行测试
            fail_to_pass = instance_info.get('FAIL_TO_PASS', [])
            pass_to_pass = instance_info.get('PASS_TO_PASS', [])
            
            if fail_to_pass or pass_to_pass:
                test_specs = {}
                if fail_to_pass:
                    test_specs['FAIL_TO_PASS'] = fail_to_pass
                if pass_to_pass:
                    test_specs['PASS_TO_PASS'] = pass_to_pass
                
                test_results = self._run_tests(test_specs, work_dir)
                
                # 处理FAIL_TO_PASS
                if 'FAIL_TO_PASS' in test_results:
                    result.fail_to_pass_results = test_results['FAIL_TO_PASS']
                    if result.fail_to_pass_results:
                        passed = sum(1 for v in result.fail_to_pass_results.values() if v)
                        result.fail_to_pass_rate = passed / len(result.fail_to_pass_results)
                
                # 处理PASS_TO_PASS
                if 'PASS_TO_PASS' in test_results:
                    result.pass_to_pass_results = test_results['PASS_TO_PASS']
                    if result.pass_to_pass_results:
                        passed = sum(1 for v in result.pass_to_pass_results.values() if v)
                        result.pass_to_pass_rate = passed / len(result.pass_to_pass_results)
            
            # 4. 判断是否成功
            fail_success = (
                not fail_to_pass or
                result.fail_to_pass_rate == 1.0
            )
            pass_success = (
                not pass_to_pass or
                result.pass_to_pass_rate == 1.0
            )
            
            result.resolved = fail_success and pass_success
        
        return result
    
    def generate_prediction_json(
        self,
        instance_id: str,
        patch_content: str,
        output_file: Optional[str] = None
    ) -> Dict[str, str]:
        """
        生成官方格式的预测JSON
        
        Args:
            instance_id: 实例ID
            patch_content: 补丁内容
            output_file: 可选的输出文件路径
        
        Returns:
            预测字典，符合官方格式
        """
        pred = {
            "instance_id": instance_id,
            "model_name_or_path": self.model_name,
            "model_patch": patch_content
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(pred, f, indent=2)
            logger.info(f"✓ Prediction saved to {output_file}")
        
        return pred
    
    def save_predictions_jsonl(
        self,
        predictions: List[Dict[str, str]],
        output_file: str
    ):
        """
        保存JSONL格式的预测文件 (官方格式)
        
        Args:
            predictions: 预测列表
            output_file: 输出文件路径
        """
        with open(output_file, 'w') as f:
            for pred in predictions:
                json.dump(pred, f)
                f.write('\n')
        
        logger.info(f"✓ Saved {len(predictions)} predictions to {output_file}")
    
    def evaluate_patches(
        self,
        patches: List[Dict[str, str]],
        instance_info: Dict[str, Any]
    ) -> EvaluationMetrics:
        """
        批量评估补丁
        
        Args:
            patches: [{"id": "...", "content": "..."}, ...]
            instance_info: 实例信息
        
        Returns:
            评估指标
        """
        instance_id = instance_info.get('instance_id', 'unknown')
        
        metrics = EvaluationMetrics(
            instance_id=instance_id,
            model_name_or_path=self.model_name,
            total_patches=len(patches)
        )
        
        for i, patch in enumerate(patches):
            patch_id = patch.get('id', f'patch_{i}')
            patch_content = patch.get('content', '')
            
            # 评估单个补丁
            result = self.evaluate_patch_lightweight(
                patch_id,
                patch_content,
                instance_info
            )
            
            metrics.patch_results.append(result)
            
            if result.applied:
                metrics.applied_patches += 1
            
            if result.resolved:
                metrics.resolved_patches += 1
        
        # 计算指标
        if metrics.total_patches > 0:
            metrics.patch_application_rate = metrics.applied_patches / metrics.total_patches
        
        if metrics.applied_patches > 0:
            metrics.success_rate = metrics.resolved_patches / metrics.applied_patches
        
        return metrics


def run_official_evaluation(
    predictions_file: str,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    max_workers: int = 1,
    run_id: str = "cgarf-eval",
    use_docker: bool = False
):
    """
    运行官方标准的SWE-Bench评估
    
    Args:
        predictions_file: 预测JSONL文件
        dataset_name: SWE-Bench数据集名称
        max_workers: 并行工作数
        run_id: 运行ID
        use_docker: 是否使用Docker (需要安装swebench)
    
    如果use_docker=True，调用官方脚本：
        python -m swebench.harness.run_evaluation \
            --predictions_path <predictions_file> \
            --dataset_name <dataset_name> \
            --max_workers <max_workers> \
            --run_id <run_id>
    """
    if use_docker:
        try:
            logger.info("[Official Docker Evaluation]")
            logger.info(f"Running official SWE-Bench evaluation with Docker...")
            logger.info(f"Command: python -m swebench.harness.run_evaluation \\")
            logger.info(f"  --predictions_path {predictions_file} \\")
            logger.info(f"  --dataset_name {dataset_name} \\")
            logger.info(f"  --max_workers {max_workers} \\")
            logger.info(f"  --run_id {run_id}")
            
            subprocess.run(
                [
                    'python3.10', '-m', 'swebench.harness.run_evaluation',
                    '--predictions_path', predictions_file,
                    '--dataset_name', dataset_name,
                    '--max_workers', str(max_workers),
                    '--run_id', run_id
                ],
                check=True
            )
            
            logger.info("✓ Official evaluation completed")
            logger.info(f"Results saved to: evaluation_results/{run_id}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Official evaluation failed: {e}")
            logger.info("Note: Official evaluation requires Docker and Python 3.10+")
    else:
        logger.info(f"[Lightweight Evaluation (Prototype)]")
        logger.info(f"Using lightweight pytest-based evaluation")
        logger.info(f"For official results, set use_docker=True")
        logger.info(f"Predictions saved to: {predictions_file}")
        logger.info(f"Command to run official evaluation:")
        logger.info(f"  python3.10 -m swebench.harness.run_evaluation \\")
        logger.info(f"    --predictions_path {predictions_file} \\")
        logger.info(f"    --dataset_name {dataset_name} \\")
        logger.info(f"    --max_workers {max_workers} \\")
        logger.info(f"    --run_id {run_id}")
