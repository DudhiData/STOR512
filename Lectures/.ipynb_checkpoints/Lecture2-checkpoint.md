# Lecture 2 
 - Fermat's Theorem:
    - Let f be a differentiable function defined on R
    - If there exists x* such that f`(x*) = 0, then x* is called a stationary point of f
    - Fermat's rule states that if x* is a local minimum or maximum of f, then f`(x*) = 0 (local extreme point)
        - A stationary point that is neither a local max or min is a saddle point
    - If f transforms its input from a multidimensional vector into a real number and is a multivariable function, then x* is a stationary point of f if the gradient of f at x* is 0

 - Motivating Example 1 (Compounding Interest Rate):
    - A nonlinear function $P(t) = P_0 * e^{Bt}$ is commonly used to model a continuous interest process of an investment over time t, where $P_0$ denotes the initial investment and B is a given interest rate
    - The parameters $P_0$ and B are unknown and we are given approximate observations of $\hat{s}(t_i)$ at several time points
    - We use the dataset to estimate the unknown parameters

 - Motivating Example 2 (Curvature Fitting):
    - Consider the function $s(t) = B_1 t + B_2 \log(t)$, where t $\exists$ [1,10] and $B_1$ and $B_2$ are two unknown parameters (decision vars)
    - Through experiments we obtain the dataset where \[hat{s}\](t) represents an approximation of s(t) at given values of t
    - We now formulate the problem of estimating the parameters $B_1$ and $B_2$ as a linear least squares problem
    - The error is $e_i = s(t_i) - \hat{s}(t_i)$
    - We minimize the sum of squared errors:
      
$$ 
\sum_{i=1}^{n} e\_i^2, \quad \text{where } n \text{ is the number of experiments.} 
$$

   \phantom{f} - Since $s(t) = B_1 t +  B_2 \log(t)$, we can write this as a least squares problem:

$$
\min_{B_1, B_2} (0.5 * \sum_{i=1}^{n} (B_1 t_i +  B_2 \log(t_i) - \hat{s}(t_i))^2
$$


