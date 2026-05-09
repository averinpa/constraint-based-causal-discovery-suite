import numpy as np
from causallearn.search.ConstraintBased.PC import pc
import citk.tests  # By importing the module, we register all our custom tests
from citk.tests import FisherZ

# =================================================================================
# Section 0: Individual Test Demonstration
# =================================================================================
print("="*30)
print("Running Individual Test Demonstration")
print("="*30)

# 1. Generate data for a simple chain: A -> B -> C
# In this structure, A and C are dependent, but they become independent when conditioned on B.
np.random.seed(1)
A = np.random.randn(200)
B = 0.8 * A + 0.2 * np.random.randn(200)
C = 0.8 * B + 0.2 * np.random.randn(200)
data_chain = np.vstack([A, B, C]).T

# 2. Instantiate a test
# We can use any test, let's use FisherZ as an example.
# We provide a cache_path to enable caching for this specific test instance.
cache_file = "examples/individual_test_cache.json"
print(f"Instantiating FisherZ with cache file: {cache_file}")
test_instance = FisherZ(data_chain, cache_path=cache_file)

# 3. Perform individual tests
print("\n--- Performing Individual CI Tests ---")
# Test A and C (should be dependent)
p_ac = test_instance(0, 2)
print(f"p-value for A _||_ C: {p_ac:.4f} -> {'Dependent' if p_ac < 0.05 else 'Independent'} (Correct: Dependent)")

# Test A and C given B (should be independent)
p_ac_b = test_instance(0, 2, [1])
print(f"p-value for A _||_ C | B: {p_ac_b:.4f} -> {'Dependent' if p_ac_b < 0.05 else 'Independent'} (Correct: Independent)")

# 4. Explicitly save the cache
# Although causal-learn saves periodically, we can call save_cache() for an immediate save.
print(f"\nExplicitly saving cache to {cache_file}")
test_instance.save_cache()


# =================================================================================
# Section 1: PC Algorithm Tests for Continuous Data
# =================================================================================
print("\n" + "="*30)
print("Running PC Algorithm Tests on Continuous Data (with caching)")
print("="*30)

# 1. Generate Continuous Data
np.random.seed(42)
# A -> C, B -> C
A_pc = np.random.randn(200)
B_pc = np.random.randn(200)
C_pc = 0.5 * A_pc + 0.5 * B_pc + 0.1 * np.random.randn(200)
data_continuous = np.vstack([A_pc, B_pc, C_pc]).T

# 2. Define the tests to run on continuous data
continuous_tests = ['fisherz', 'spearman', 'dcor', 'kci', 'reg', 'rf', 'dml', 'crit', 'edml']
print(f"PC algorithm runs will use a separate cache file for each test.\n")

# 3. Run PC Algorithm for each test
for test_name in continuous_tests:
    print(f"\n--- Running PC with '{test_name}' ---")
    try:
        cg = pc(data_continuous, alpha=0.05, indep_test=test_name, cache_path=f"examples/continuous_{test_name}_cache.json")
        print("Learned Graph:")
        print(cg.G)
    except Exception as e:
        print(f"Error running {test_name}: {e}")

# =================================================================================
# Section 2: Tests for Discrete (Categorical) Data
# =================================================================================
print("\n" + "="*30)
print("Running Tests on Discrete Data")
print("="*30)

# 1. Generate Discrete Data
# X -> Z, Y -> Z
X_disc = np.random.randint(0, 3, size=500)
Y_disc = np.random.randint(0, 3, size=500)
Z_disc = (X_disc + Y_disc + np.random.randint(0, 2, size=500)) % 3
data_discrete = np.vstack([X_disc, Y_disc, Z_disc]).T

# 2. Define the tests to run on discrete data
discrete_tests = ['chisq', 'gsq']

# 3. Run PC Algorithm for each test
for test_name in discrete_tests:
    print(f"\n--- Running PC with '{test_name}' ---")
    try:
        cg = pc(data_discrete, alpha=0.05, indep_test=test_name, cache_path="examples/discrete_cache.json")
        print("Learned Graph:")
        print(cg.G)
    except Exception as e:
        print(f"Error running {test_name}: {e}")

# =================================================================================
# Section 3: Test for Discrete (Binary) Data
# =================================================================================
print("\n" + "="*30)
print("Running Test on Binary Data")
print("="*30)

# 1. Generate Binary Data
# X -> Z, Y -> Z
np.random.seed(123)
X_bin = np.random.binomial(1, 0.5, 500)
Y_bin = np.random.binomial(1, 0.5, 500)
log_odds = 1.5 * X_bin - 1.5 * Y_bin
prob = 1 / (1 + np.exp(-log_odds))
Z_bin = np.random.binomial(1, prob)
data_binary = np.vstack([X_bin, Y_bin, Z_bin]).T

# 2. Define and Run PC Algorithm with logit, gsq, and chisq tests
binary_tests = ['logit', 'gsq', 'chisq']
for test_name in binary_tests:
    print(f"\n--- Running PC with '{test_name}' ---")
    try:
        cg = pc(data_binary, alpha=0.05, indep_test=test_name, cache_path=f"examples/binary_{test_name}_cache.json")
        print("Learned Graph:")
        print(cg.G)
    except Exception as e:
        print(f"Error running {test_name}: {e}")

# =================================================================================
# Section 4: Test for Discrete (Count) Data
# =================================================================================
print("\n" + "="*30)
print("Running Test on Count Data")
print("="*30)

# 1. Generate Count Data
# X -> Z, Y -> Z
np.random.seed(456)
X_count = np.random.randint(0, 5, 500)
Y_count = np.random.randint(0, 5, 500)
rate = np.exp(0.5 * X_count + 0.5 * Y_count - 1)
Z_count = np.random.poisson(rate)
data_count = np.vstack([X_count, Y_count, Z_count]).T

# 2. Run PC Algorithm with pois test
print(f"\n--- Running PC with 'pois' ---")
try:
    cg = pc(data_count, alpha=0.05, indep_test='pois', cache_path="examples/count_cache.json")
    print("Learned Graph:")
    print(cg.G)
except Exception as e:
    print(f"Error running pois: {e}")
