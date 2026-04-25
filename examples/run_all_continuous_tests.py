import numpy as np
import os

from citk.tests import FisherZ, Spearman


def run_tests_on_dataset(data, dataset_name):
    """Helper function to run all tests on a given dataset."""
    print("\n" + "="*60)
    print(f"Running All Continuous CI Tests on: {dataset_name}")
    print(f"Data shape: {data.shape}")
    print("="*60)

    all_continuous_tests = [
        (FisherZ, "Fisher's Z", {}, True),
        (Spearman, "Spearman's Rho", {}, True),
    ]

    for TestClass, test_name, kwargs, supports_conditional in all_continuous_tests:
        print(f"\n--- Testing: {test_name} ({dataset_name}) ---")

        cache_file = f"examples/cache/{dataset_name}_{test_name.lower().replace(' ', '_')}_cache.json"
        test_instance = TestClass(data, cache_path=cache_file, **kwargs)

        # Unconditional test
        p_ac = test_instance(0, 2)
        print(f"  p-value for A _||_ C: {p_ac:.4f} -> {'Dependent' if p_ac < 0.05 else 'Independent'} (Expected: Dependent)")

        # Conditional test
        if supports_conditional:
            p_ac_b = test_instance(0, 2, [1])
            print(f"  p-value for A _||_ C | B: {p_ac_b:.4f} -> {'Dependent' if p_ac_b < 0.05 else 'Independent'} (Expected: Independent)")
        else:
            print("  p-value for A _||_ C | B: Not supported by this test.")
        
        test_instance.save_cache()


# =================================================================================
# Section 1: Low Sample Size / Weak Signal Data
# =================================================================================
np.random.seed(1)
A_weak = np.random.randn(200)
B_weak = 0.8 * A_weak + 0.2 * np.random.randn(200)
C_weak = 0.8 * B_weak + 0.2 * np.random.randn(200)
data_weak = np.vstack([A_weak, B_weak, C_weak]).T

# Create a directory for cache files if it doesn't exist
os.makedirs("examples/cache", exist_ok=True)

run_tests_on_dataset(data_weak, "Weak Signal (n=200)")

# =================================================================================
# Section 2: High Sample Size / Strong Signal Data
# =================================================================================
np.random.seed(1)
A_strong = np.random.randn(500)
B_strong = 0.9 * A_strong + 0.1 * np.random.randn(500)
C_strong = 0.9 * B_strong + 0.1 * np.random.randn(500)
data_strong = np.vstack([A_strong, B_strong, C_strong]).T

run_tests_on_dataset(data_strong, "Strong Signal (n=500)")

print("\n" + "="*60)
print("Demonstration complete.")
print("="*60) 