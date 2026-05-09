# Interpreting Test Results

After running a conditional independence (CI) test, the primary output you will work with is the **p-value**. Understanding what this value represents is key to correctly interpreting the outcome of your analysis.

## The P-Value

A CI test starts with a **null hypothesis** ($H_0$), which states that the two variables of interest, X and Y, **are** conditionally independent given the conditioning set Z.

$H_0: X \perp Y \mid Z$

The p-value is the probability of observing a test statistic at least as extreme as the one computed from your data, *assuming the null hypothesis is true*.

### How to Interpret It

- **High P-Value (e.g., > 0.05)**: A high p-value suggests that your data is consistent with the null hypothesis. You **fail to reject** the null hypothesis and conclude that there is not enough evidence to say that X and Y are conditionally dependent. For the purposes of causal discovery algorithms like PC, this is treated as evidence for conditional independence.

- **Low P-Value (e.g., <= 0.05)**: A low p-value suggests that your data is unlikely to have occurred if the null hypothesis were true. You **reject** the null hypothesis and conclude that X and Y are **conditionally dependent** given Z.

The threshold used to make this decision (commonly 0.05 or 0.01) is known as the **significance level** (alpha). This is a parameter you set when running algorithms like PC.

## Accessing the Results

All tests in `citk` that are compatible with `causal-learn` are designed to be called directly by the search algorithm (like `pc`). The algorithm handles the interpretation internally.

When you call a test yourself, it returns the p-value directly.

```python
import numpy as np
from citk.tests import FisherZ

# Generate some dependent data
data = np.random.randn(200, 2)
data[:, 1] = 2 * data[:, 0] + 0.1 * np.random.randn(200)

# Initialize and run the test
test = FisherZ(data)
p_value = test(0, 1)

print(f"P-value: {p_value:.4f}")

if p_value <= 0.05:
    print("Conclusion: The variables are dependent.")
else:
    print("Conclusion: We cannot reject that the variables are independent.")
``` 