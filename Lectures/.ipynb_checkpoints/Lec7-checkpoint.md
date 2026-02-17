 # Lecture 7 (Nonlinear Least Squares)

 - Modeled as $y = f(x; \beta) + \epsilon$, wher e the parameter $\beta$ enters the model function in a nonlinear manner and may consist of multiple components
 - Generally more challenging and may require iterative numerical methods

- Motivating Example:
    - In fisheries biology, relationship between spawning stock size S and resulting number of fish is modeled using $y = f(S; \beta) + \epsilon$
        - Where $f(S: \beta) = \frac{\alpha S}{1 + \frac{S}{k}}$ and $\beta = (\alpha, k)^T$ and $\epsilon$ is a random noise term with 0 mean
        - The parameters $\alpha$ and k are unknown model parameters that must be estimated from data and collected into $\beta$
        - $\alpha$ is the slope of the recruitment function at S = 0
        - $\alpha k$ is the maximum recruitment level
        - k is the stock size at which recruitment reaches half of its maximum value
    - Assume that the noise follows a Gaussian distribution with 0 mean and variance $\sigma^2$ --> under this assumption, estimating $\beta$ by maximum log-likelihood is equivalent to minimizing the sum of squared residuals
    - $E(\beta) = \sum_{i = 1}^{n} \epsilon^2 = \sum_{i = 1}^{n} (y_i - f(x_i; \beta))^2$
    - The residual vector is $F(\beta) = (f(x_1; \beta) - y_1, \dots, f(x_n; \beta) - y_n)
    - Thus, the objective function is:

$$
L(\beta) = \frac{1}{2} \lVert F(\beta) \rVert_2^2 = \frac{1}{2} \sum_{i = 1}^{n} (f(x_i; \beta) - y_i)^2
$$

- Therefore, the nonlinear least-squares problem:

$$
min_{\beta \exists \mathbb{R}^p} { L(\beta) = \frac{1}{2} \lVert F(\beta) \rVert_2^2 }
$$

- Fermat's rule for this problem is $\nabla L(\beta_*) = 0$ <=> $F'(\beta_*)^TF(\beta_*) = 0$
    - Where $F'(\beta)$ denotes the Jacobian matrix of F at $\beta$
 
- Unlike linear least-squares, L may be nonconvex, which means the vector satisfying Fermat's might be a local minimizer, a saddle point, or even a local maximizer

    -   