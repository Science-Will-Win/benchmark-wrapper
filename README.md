# AIGEN BioAgent - Benchmark Wrapper

AIGEN Sciences × Modulabs 프로젝트의 생의학 AI 에이전트 벤치마크 평가 래퍼

## 개요

3개 벤치마크(Biomni-Eval1, LAB-Bench, BioML-Bench)를 통합 인터페이스로 평가하는 래퍼입니다.

## 지원 벤치마크

| 벤치마크 | 태스크 수 | 소스 | 형식 |
|----------|----------|------|------|
| Biomni-Eval1 | 433개 (12개 서브셋) | HuggingFace `biomni/Eval1` | 자유 응답 |
| LAB-Bench | 1,967개 (8개 카테고리) | HuggingFace `futurehouse/lab-bench` | 객관식 |
| BioML-Bench | 24개 (4개 도메인) | GitHub | ML 파이프라인 |

## 파일 구조
```
benchmark_wrapper.py   # 메인 래퍼 (BiomniEval1Wrapper, LabBenchWrapper)
wrapper_test.py        # 래퍼 기본 테스트
test_benchmark.py      # 벤치마크 연결 테스트
```

## 사용법
```python
from benchmark_wrapper import BiomniEval1Wrapper, LabBenchWrapper, AgentResponse

# Biomni-Eval1
wrapper = BiomniEval1Wrapper()
wrapper.load_tasks(limit=10)

# 에이전트 응답 평가


