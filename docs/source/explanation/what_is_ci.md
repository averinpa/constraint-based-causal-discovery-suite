# What is Conditional Independence?

Conditional Independence (CI) is a fundamental concept in probability theory and statistics, forming the bedrock of modern causal inference and graphical modeling.

## Formal Definition

Two variables, $X$ and $Y$, are said to be conditionally independent of each other given a third variable (or set of variables), $Z$, if and only if their conditional joint distribution, given $Z$, is equal to the product of their individual conditional distributions given $Z$.

Mathematically, this is written as:

$X \perp Y \mid Z \iff P(X, Y \mid Z) = P(X \mid Z)P(Y \mid Z)$

This statement implies that once we know the value of $Z$, gaining knowledge about $Y$ tells us nothing new about $X$, and vice-versa. The conditioning set $Z$ "blocks" the flow of information between $X$ and $Y$.

## Intuitive Explanation

Consider the following variables:
- **X**: Ice Cream Sales
- **Y**: Number of Drownings
- **Z**: Average Daily Temperature

In a typical city, you would observe a strong positive correlation between ice cream sales and drownings. As one goes up, the other tends to go up.

However, this relationship is not causal. It is a result of a **common cause**, or a **confounder**: the temperature.

- When the temperature is high, more people buy ice cream.
- When the temperature is high, more people go swimming, which unfortunately leads to more drownings.

If we **condition** on temperature (i.e., we look at data from only the days where the temperature was, say, 75°F), the relationship between ice cream sales and drownings would vanish. Knowing how many ice creams were sold on a 75°F day gives you no new information about how many drownings occurred on that same day.

Thus, we can say: **Ice Cream Sales $\perp$ Drownings $\mid$ Temperature**.

## Why It Matters in Causal Discovery

Conditional independence tests are the primary tool used by constraint-based causal discovery algorithms (like the PC algorithm). These algorithms work by systematically performing CI tests on the data to identify the "d-separations" in the underlying causal graph.

By determining which variables become independent when conditioned on others, these algorithms can prune edges from a fully connected graph to reveal the likely causal structure that generated the data. 