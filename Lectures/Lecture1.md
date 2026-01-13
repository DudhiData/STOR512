# Lecture 1 (Intro to Optimization)
 - A mathematical optimization problem has 3 components:
    1. One or more multiple decision variables (model parameters)
        - Unknown quantities that have to be determined (e.g. scalars, vectors, matrices, functions)
    2. At most one objective function (if you have many --> combine them)
        - Function of the decision variables (outputs in real #s)
        - Example: in portfolio optimization we try to maximize return and minimize risk f(w):=R(w)-xG(w) where x is a weight           parameter to balance risk and return and R and G are risk and return respectively taking a portfolio vector w
    3. Set of constraints
        - Requirements imposed on decision variables; when none exist the problem becomes an unconstrained one

 - The problem aims to determine values of the decision variables that optimize an objective function given a set of constraints

Mathematical Form:
 - min f(x) where x exists in the set of constraints
 - max f(x) where x exists in the set of constraints
 - The min (respectively, the max) notation means that for all values of decision variables x that satisfy all constraints in    F, we have x⋆∈F such that f(x⋆) ≤ f(x) (respectively, f(x⋆) ≥ f(x)) for all x ∈ F (global optimal solution)