"""
AIGEN BioAgent - 통합 벤치마크 래퍼
=====================================
Phase 1: Biomni-Eval1, LAB-Bench, BioML-Bench 통합 인터페이스

작성일: 2026-02-05
담당자: Hoony (Phase 1 리드)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datasets import load_dataset
import time


# ============================================
# 1. 공통 데이터 구조
# ============================================

class BenchmarkType(Enum):
    """벤치마크 종류"""
    BIOMNI_EVAL1 = "biomni_eval1"
    LAB_BENCH = "lab_bench"
    BIOML_BENCH = "bioml_bench"


@dataclass
class BenchmarkTask:
    """벤치마크 태스크 공통 구조"""
    task_id: str                    # 고유 ID
    benchmark: BenchmarkType        # 벤치마크 종류
    question: str                   # 질문/프롬프트
    ground_truth: Any               # 정답
    task_type: str                  # 태스크 유형 (예: LitQA, ProtocolQA 등)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 추가 정보


@dataclass
class AgentResponse:
    """에이전트 응답 구조"""
    task_id: str                    # 태스크 ID
    answer: Any                     # 에이전트 답변
    trajectory: List[Dict] = field(default_factory=list)  # 실행 과정 로그
    latency_ms: float = 0.0         # 응답 시간 (밀리초)
    token_usage: Dict[str, int] = field(default_factory=dict)  # 토큰 사용량
    errors: List[str] = field(default_factory=list)  # 발생한 에러들


@dataclass
class EvaluationResult:
    """평가 결과 구조"""
    task_id: str                    # 태스크 ID
    score: float                    # 점수 (0.0 ~ 1.0)
    is_correct: bool                # 정답 여부
    ground_truth: Any               # 정답
    predicted: Any                  # 예측값
    details: Dict[str, Any] = field(default_factory=dict)  # 상세 정보


# ============================================
# 2. 추상 베이스 클래스
# ============================================

class BaseBenchmarkWrapper(ABC):
    """벤치마크 래퍼 추상 베이스 클래스"""
    
    def __init__(self):
        self.tasks: List[BenchmarkTask] = []
        self.loaded = False
    
    @property
    @abstractmethod
    def benchmark_type(self) -> BenchmarkType:
        """벤치마크 종류 반환"""
        pass
    
    @property
    @abstractmethod
    def benchmark_name(self) -> str:
        """벤치마크 이름 반환"""
        pass
    
    @abstractmethod
    def load_tasks(self, **kwargs) -> List[BenchmarkTask]:
        """태스크 로딩"""
        pass
    
    @abstractmethod
    def evaluate(self, task: BenchmarkTask, response: AgentResponse) -> EvaluationResult:
        """단일 태스크 평가"""
        pass
    
    def evaluate_batch(self, responses: List[AgentResponse]) -> List[EvaluationResult]:
        """배치 평가"""
        task_map = {t.task_id: t for t in self.tasks}
        results = []
        for resp in responses:
            if resp.task_id in task_map:
                result = self.evaluate(task_map[resp.task_id], resp)
                results.append(result)
        return results
    
    def get_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """평가 결과 요약"""
        if not results:
            return {"total": 0, "accuracy": 0.0, "average_score": 0.0}
        
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        avg_score = sum(r.score for r in results) / total
        
        return {
            "benchmark": self.benchmark_name,
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
            "average_score": avg_score
        }


# ============================================
# 3. Biomni-Eval1 래퍼
# ============================================

class BiomniEval1Wrapper(BaseBenchmarkWrapper):
    """Biomni-Eval1 벤치마크 래퍼"""
    
    DATASET_PATH = "biomni/Eval1"
    SPLIT = "test"  # 문서에는 train이지만 실제는 test
    
    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.BIOMNI_EVAL1
    
    @property
    def benchmark_name(self) -> str:
        return "Biomni-Eval1"
    
    def load_tasks(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        """
        Biomni-Eval1 태스크 로딩
        
        Args:
            limit: 로딩할 최대 태스크 수 (None이면 전체)
        
        Returns:
            BenchmarkTask 리스트
        """
        print(f"📥 {self.benchmark_name} 로딩 중...")
        dataset = load_dataset(self.DATASET_PATH, split=self.SPLIT)
        
        self.tasks = []
        for i, item in enumerate(dataset):
            if limit and i >= limit:
                break
            
            task = BenchmarkTask(
                task_id=item["instance_id"],
                benchmark=self.benchmark_type,
                question=item["prompt"],
                ground_truth=item["answer"],
                task_type=item["task_name"],
                metadata={
                    "task_instance_id": item["task_instance_id"],
                    "split": item["split"]
                }
            )
            self.tasks.append(task)
        
        self.loaded = True
        print(f"✅ {len(self.tasks)}개 태스크 로딩 완료")
        return self.tasks
    
    def evaluate(self, task: BenchmarkTask, response: AgentResponse) -> EvaluationResult:
        """
        Biomni-Eval1 평가: Exact Match
        """
        predicted = str(response.answer).strip().lower()
        ground_truth = str(task.ground_truth).strip().lower()
        
        is_correct = predicted == ground_truth
        score = 1.0 if is_correct else 0.0
        
        return EvaluationResult(
            task_id=task.task_id,
            score=score,
            is_correct=is_correct,
            ground_truth=task.ground_truth,
            predicted=response.answer,
            details={
                "evaluation_method": "exact_match",
                "task_type": task.task_type
            }
        )


# ============================================
# 4. LAB-Bench 래퍼
# ============================================

class LabBenchWrapper(BaseBenchmarkWrapper):
    """LAB-Bench 벤치마크 래퍼"""
    
    DATASET_PATH = "futurehouse/lab-bench"
    SPLIT = "train"  # 문서에는 test이지만 실제는 train
    
    # 사용 가능한 카테고리
    CATEGORIES = [
        "CloningScenarios", "DbQA", "FigQA", "LitQA2",
        "ProtocolQA", "SeqQA", "SuppQA", "TableQA"
    ]
    
    def __init__(self, categories: Optional[List[str]] = None):
        """
        Args:
            categories: 로딩할 카테고리 리스트 (None이면 전체)
        """
        super().__init__()
        self.selected_categories = categories or self.CATEGORIES
    
    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.LAB_BENCH
    
    @property
    def benchmark_name(self) -> str:
        return "LAB-Bench"
    
    def load_tasks(self, limit_per_category: Optional[int] = None) -> List[BenchmarkTask]:
        """
        LAB-Bench 태스크 로딩
        
        Args:
            limit_per_category: 카테고리당 최대 태스크 수
        
        Returns:
            BenchmarkTask 리스트
        """
        print(f"📥 {self.benchmark_name} 로딩 중...")
        self.tasks = []
        
        for category in self.selected_categories:
            try:
                print(f"   - {category} 로딩 중...")
                dataset = load_dataset(self.DATASET_PATH, category, split=self.SPLIT)
                
                for i, item in enumerate(dataset):
                    if limit_per_category and i >= limit_per_category:
                        break
                    
                    # 선택지 구성: ideal + distractors
                    options = [item["ideal"]] + item["distractors"]
                    
                    task = BenchmarkTask(
                        task_id=item["id"],
                        benchmark=self.benchmark_type,
                        question=item["question"],
                        ground_truth=item["ideal"],  # 정답은 ideal
                        task_type=category,
                        metadata={
                            "options": options,
                            "distractors": item["distractors"],
                            "source": item.get("source", ""),
                            "subtask": item.get("subtask", ""),
                            "is_opensource": item.get("is_opensource", False)
                        }
                    )
                    self.tasks.append(task)
                    
            except Exception as e:
                print(f"   ⚠️ {category} 로딩 실패: {e}")
        
        self.loaded = True
        print(f"✅ {len(self.tasks)}개 태스크 로딩 완료")
        return self.tasks
    
    def evaluate(self, task: BenchmarkTask, response: AgentResponse) -> EvaluationResult:
        """
        LAB-Bench 평가: 정답(ideal)과 매칭
        """
        predicted = str(response.answer).strip()
        ground_truth = str(task.ground_truth).strip()
        
        # 정확히 일치하거나, 정답이 응답에 포함되어 있으면 정답
        is_correct = (predicted.lower() == ground_truth.lower() or 
                      ground_truth.lower() in predicted.lower())
        score = 1.0 if is_correct else 0.0
        
        return EvaluationResult(
            task_id=task.task_id,
            score=score,
            is_correct=is_correct,
            ground_truth=task.ground_truth,
            predicted=response.answer,
            details={
                "evaluation_method": "ideal_match",
                "task_type": task.task_type,
                "options": task.metadata.get("options", [])
            }
        )


# ============================================
# 5. BioML-Bench 래퍼 (기본 구조)
# ============================================

class BioMLBenchWrapper(BaseBenchmarkWrapper):
    """BioML-Bench 벤치마크 래퍼 (GitHub 기반)"""
    
    GITHUB_REPO = "https://github.com/BioML-bench/bioml-bench"
    
    # 24개 태스크 도메인
    DOMAINS = [
        "protein_engineering",
        "drug_discovery", 
        "genomics",
        "clinical_prediction"
    ]
    
    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.BIOML_BENCH
    
    @property
    def benchmark_name(self) -> str:
        return "BioML-Bench"
    
    def load_tasks(self, **kwargs) -> List[BenchmarkTask]:
        """
        BioML-Bench 태스크 로딩
        
        Note: GitHub 기반이라 별도 다운로드 필요
        현재는 플레이스홀더 구현
        """
        print(f"📥 {self.benchmark_name} 로딩 중...")
        print(f"   ⚠️ GitHub 기반 벤치마크 - 별도 설정 필요")
        print(f"   📎 저장소: {self.GITHUB_REPO}")
        
        # 플레이스홀더: 실제 구현 시 GitHub에서 데이터 로딩
        self.tasks = []
        self.loaded = True
        
        return self.tasks
    
    def evaluate(self, task: BenchmarkTask, response: AgentResponse) -> EvaluationResult:
        """
        BioML-Bench 평가: ML 메트릭 기반 (AUROC, Spearman 등)
        """
        # 플레이스홀더: 실제 구현 시 태스크별 메트릭 적용
        return EvaluationResult(
            task_id=task.task_id,
            score=0.0,
            is_correct=False,
            ground_truth=task.ground_truth,
            predicted=response.answer,
            details={
                "evaluation_method": "ml_metrics",
                "note": "BioML-Bench 평가는 태스크별 ML 메트릭 필요"
            }
        )


# ============================================
# 6. 통합 벤치마크 매니저
# ============================================

class BenchmarkManager:
    """벤치마크 통합 관리자"""
    
    def __init__(self):
        self.wrappers: Dict[BenchmarkType, BaseBenchmarkWrapper] = {}
    
    def register(self, wrapper: BaseBenchmarkWrapper):
        """래퍼 등록"""
        self.wrappers[wrapper.benchmark_type] = wrapper
        print(f"📌 {wrapper.benchmark_name} 등록됨")
    
    def load_all(self, **kwargs):
        """모든 벤치마크 로딩"""
        for wrapper in self.wrappers.values():
            wrapper.load_tasks(**kwargs)
    
    def get_all_tasks(self) -> List[BenchmarkTask]:
        """모든 태스크 반환"""
        all_tasks = []
        for wrapper in self.wrappers.values():
            all_tasks.extend(wrapper.tasks)
        return all_tasks
    
    def evaluate_all(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """모든 벤치마크 평가"""
        results = {}
        
        # 응답을 벤치마크별로 분류
        for wrapper in self.wrappers.values():
            task_ids = {t.task_id for t in wrapper.tasks}
            relevant_responses = [r for r in responses if r.task_id in task_ids]
            
            if relevant_responses:
                eval_results = wrapper.evaluate_batch(relevant_responses)
                results[wrapper.benchmark_name] = wrapper.get_summary(eval_results)
        
        return results


# ============================================
# 7. 테스트 코드
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 벤치마크 래퍼 테스트")
    print("=" * 60)
    
    # 1. Biomni-Eval1 테스트
    print("\n[1] Biomni-Eval1 래퍼 테스트")
    biomni = BiomniEval1Wrapper()
    tasks = biomni.load_tasks(limit=5)
    
    print(f"\n   샘플 태스크:")
    print(f"   - ID: {tasks[0].task_id}")
    print(f"   - 타입: {tasks[0].task_type}")
    print(f"   - 질문: {tasks[0].question[:100]}...")
    print(f"   - 정답: {tasks[0].ground_truth}")
    
    # 평가 테스트 (가상 응답)
    fake_response = AgentResponse(
        task_id=tasks[0].task_id,
        answer=tasks[0].ground_truth  # 정답으로 테스트
    )
    result = biomni.evaluate(tasks[0], fake_response)
    print(f"\n   평가 결과: score={result.score}, correct={result.is_correct}")
    
    # 2. LAB-Bench 테스트
    print("\n" + "-" * 60)
    print("[2] LAB-Bench 래퍼 테스트")
    labbench = LabBenchWrapper(categories=["LitQA2"])
    tasks = labbench.load_tasks(limit_per_category=5)
    
    print(f"\n   샘플 태스크:")
    print(f"   - ID: {tasks[0].task_id}")
    print(f"   - 타입: {tasks[0].task_type}")
    print(f"   - 질문: {tasks[0].question[:100]}...")
    print(f"   - 정답: {tasks[0].ground_truth[:50]}...")
    
    # 3. 통합 매니저 테스트
    print("\n" + "-" * 60)
    print("[3] 통합 매니저 테스트")
    manager = BenchmarkManager()
    manager.register(BiomniEval1Wrapper())
    manager.register(LabBenchWrapper(categories=["LitQA2"]))
    
    print("\n" + "=" * 60)
    print("✅ 래퍼 테스트 완료!")
    print("=" * 60)
