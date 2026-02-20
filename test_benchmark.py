from datasets import load_dataset

print("=" * 50)
print("🔍 벤치마크 데이터 접근 테스트")
print("=" * 50)

# 1. Biomni-Eval1 (split: test)
print("\n[1] Biomni-Eval1 테스트...")
try:
    dataset = load_dataset("biomni/Eval1", split="test")
    print(f"   ✅ 성공! 샘플 수: {len(dataset)}")
    print(f"   📋 키: {list(dataset[0].keys())}")
except Exception as e:
    print(f"   ❌ 실패: {e}")

# 2. LAB-Bench (config 지정 필요)
print("\n[2] LAB-Bench 테스트...")
try:
    dataset = load_dataset("futurehouse/lab-bench", "LitQA2", split="train")
    print(f"   ✅ 성공! 샘플 수: {len(dataset)}")
    print(f"   📋 키: {list(dataset[0].keys())}")
except Exception as e:
    print(f"   ❌ 실패: {e}")

print("\n" + "=" * 50)
print("✅ 서버 환경 벤치마크 접근 테스트 완료!")
print("=" * 50)
