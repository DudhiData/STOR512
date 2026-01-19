# Lecture 2 
 - Fermat's Theorem:
    - Let f be a differentiable function defined on R
    - If there exists x* such that f`(x*) = 0, then x* is called a stationary point of f
    - Fermat's rule states that if x* is a local minimum or maximum of f, then f`(x*) = 0 (local extreme point)
        - A stationary point that is neither a local max or min is a saddle point
    - If f transforms its input from a multidimensional vector into a real number and is a multivariable function, then x* is a stationary point of f if the gradient of f at x* is 0

 - Motivating Example 1 (Compounding Interest Rate):
    - A nonlinear function P(t) = P~0~ * e^Bt^ is commonly used to model a continuous interest process of an investment over time t, where P~0~ denotes the initial investment and B is a given interest rate
    - The parameters P~0~ and B are unknown and we are given approximate observations of P^(t~i~) at several time points
    - We use the dataset to estimate the unknown parameters

 - Motivating Example 2 (Curvature Fitting):
    - Consider the function s(t) = B~1~t + B~2~log(t), where t $\exists$ [1,10] and B~1~ and B~2~ are two unknown parameters (decision vars)
    - Through experiments we obtain the dataset where s^(t) represents an approximation of s(t) at given values of t
    - We now formulate the problem of estimating the parameters B~1~ and B~2~ as a linear least squares problem
    - The error between s^(t) and s(t) is e(t) = s(t) - s^(t)
    - We minimize the sum of squared errors $\sum_{i=1}^{n} e(t)^2^$ where n is our # of experiments
    - Since s(t) = B~1~t + B~2~log(t), we can write this as a least squares problem:
        - min (B~1~, B~2~) {1/2 * $\sum_{i=1}^{n} (B~1~*t~i~ + B~2~*log(t~i~) - s^(t~i~))^2^$}