# Lecture 1 (Intro to Optimization)
- A mathematical optimization problem has 3 components:
    1. One or more multiple decision variables (model parameters)
        - Unknown quantities that have to be determined (e.g. scalars, vectors, matrices, functions)
    2. At most one objective function (if you have many --> combine them)
        - Function of the decision variables (outputs in real #s)
        - Example: in portfolio optimization we try to maximize return and minimize risk f(w):=R(w)-xG(w) where x is a weight parameter to balance risk and return and R and G are risk and return respectively, taking a portfolio vector w
    3. Set of constraints
        - Requirements imposed on decision variables; when none exist, the problem becomes an unconstrained one

- The problem aims to determine values of the decision variables that optimize an objective function given a set of constraints

- Mathematical Form:
 - min f(x) where x exists in the set of constraints
 - max f(x) where x exists in the set of constraints
 - The min (respectively, the max) notation means that for all values of decision variables x that satisfy all constraints in F, we have x⋆∈F such that f(x⋆) ≤ f(x) (respectively, f(x⋆) ≥ f(x)) for all x ∈ F (global optimal solution)

 - Example: Linear Regression
      - Find best model parameters B ∈ R^d^ of a linear regression model y = X^T^B such that it minimizes the least-squares loss function
  
- Feasible Solutions:
    - A vector x is a feasible solution if it satisfies all constraints in F
    - The set of all feasible solutions is called the feasible set denoted by F

- Optimal Solutions:
    - A feasible solution is an optimal solution if f(x*) <= f(x) for all x ∈ F in the minimization problem (f(x*) >= f(x) for maximization)
    - Every optimal solution gives us the same optimal value (value of objective function at x*)

- Other Concepts:
    - Local Optima: an optimal solution x* is called a local optimal solution if it gives us the best objective value in a local neighborhood around x*
    - Global Optima: if this neighborhood is the entire feasible set F, then x⋆ is called a global optimal solution