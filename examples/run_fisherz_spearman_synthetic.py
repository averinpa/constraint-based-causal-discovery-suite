import numpy as np
import os
import time

# By importing the citk.tests module, we ensure that all custom tests
# defined within it are registered with causal-learn.
import citk.tests
from citk.tests.simple_tests import FisherZ, Spearman

# Import the PC algorithm from causal-learn
from causallearn.search.ConstraintBased.PC import pc


def test_individual_ci(TestClass, data, dataset_name):
    """
    Tests a given CI test class individually to confirm its correctness
    and demonstrate explicit cache saving and loading.
    """
    test_name = TestClass.__name__
    print("\n" + "="*80)
    print(f"Part 1: Running Individual Test for: {test_name} on {dataset_name}")
    print("="*80)

    cache_path = f"examples/cache/individual_{test_name.lower()}_cache.json"
    if os.path.exists(cache_path):
        os.remove(cache_path)

    # --- First Run: Compute and save p-values ---
    print("\n--- Running initial tests (populating cache)...")
    instance1 = TestClass(data, cache_path=cache_path)
    
    p_ac = instance1(0, 2)  # A vs C, should be dependent
    print(f"  p-value for A _||_ C: {p_ac:.4f} -> {'Dependent' if p_ac < 0.05 else 'Independent'} (Expected: Dependent)")
    
    p_ac_b = instance1(0, 2, [1])  # A vs C given B, should be independent
    print(f"  p-value for A _||_ C | B: {p_ac_b:.4f} -> {'Dependent' if p_ac_b < 0.05 else 'Independent'} (Expected: Independent)")

    # Explicitly save the cache.
    instance1.save_cache()
    print(f"Cache explicitly saved to {cache_path}")

    # --- Second Run: Load and verify p-values from cache ---
    print("\n--- Rerunning tests (loading from cache)...")
    start_time = time.time()
    instance2 = TestClass(data, cache_path=cache_path) # Loads cache on init
    p_ac_cached = instance2(0, 2)
    load_time = time.time() - start_time
    
    print(f"  Took {load_time:.6f} seconds to load and retrieve from cache.")
    print(f"  Cached p-value matches original: {p_ac == p_ac_cached}")


def test_pc_algorithm_integration(test_id, test_name, data, dataset_name):
    """
    Tests the integration of a CI test within the PC algorithm, demonstrating
    the automatic caching functionality provided by the __del__ method.
    """
    print("\n" + "="*80)
    print(f"Part 2: Testing PC Algorithm Integration for: {test_name} on {dataset_name}")
    print("="*80)

    cache_path = f"examples/cache/pc_{test_id}_{dataset_name}_cache.json"
    if os.path.exists(cache_path):
        os.remove(cache_path)

    # --- First Run: PC populates the cache, which is auto-saved by __del__ ---
    print(f"\n--- Running PC with {test_name} (1st time)...")
    start_time = time.time()
    cg_first_run = pc(data, alpha=0.05, indep_test=test_id, cache_path=cache_path)
    first_run_time = time.time() - start_time
    print(f"  First run took: {first_run_time:.4f} seconds.")

    # --- Second Run: PC should be faster as it uses the auto-saved cache ---
    print(f"\n--- Running PC with {test_name} (2nd time)...")
    start_time = time.time()
    pc(data, alpha=0.05, indep_test=test_id, cache_path=cache_path)
    second_run_time = time.time() - start_time
    print(f"  Second run took: {second_run_time:.4f} seconds.")
    
    if second_run_time < first_run_time * 0.5:
        print("\n  SUCCESS: Automatic caching via __del__ is working as expected.")
    else:
        print("\n  INFO: Caching test inconclusive. Second run was not significantly faster.")

    print("\n  Learned Graph Edges (from first run):")
    # The expected graph is A-B-C. PC finds the skeleton but cannot orient the edges.


# =================================================================================
# Main Execution Block
# =================================================================================

# 1. Generate a synthetic dataset with a clear causal structure: A -> B -> C
print("Generating synthetic data (A -> B -> C)...")
np.random.seed(42)
n_samples = 500
A = np.random.randn(n_samples)
B = 0.8 * A + 0.3 * np.random.randn(n_samples)
C = 0.8 * B + 0.3 * np.random.randn(n_samples)
dataset = np.vstack([A, B, C]).T
dataset_name = "synthetic_chain"

# 2. Ensure the cache directory exists
os.makedirs("examples/cache", exist_ok=True)

# 3. Run the tests for FisherZ
test_individual_ci(FisherZ, dataset, dataset_name)
test_pc_algorithm_integration("fisherz_citk", "FisherZ", dataset, dataset_name)

# 4. Run the tests for Spearman's Rho
test_individual_ci(Spearman, dataset, dataset_name)
test_pc_algorithm_integration("spearman", "Spearman's Rho", dataset, dataset_name)

print("\n" + "="*80)
print("Comprehensive CITK Test complete.")
print("="*80)
