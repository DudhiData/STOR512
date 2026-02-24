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
    -   We need to verify that $L(\beta_*) \leq L(\beta) \quad \forall \beta \exists \mathbb{R}^p$
 
- Gauss-Newton Method Initialization:
    - Start from initial point $\beta_0$, at each iteration $t \geq 0$ we approximate the objective locally by linearizing the residual function $F$ around $\beta_t$
    - A Newton-type linearization of the gradient:
        - $\nabla^2L(\beta_t)(\beta - \beta_t) + \nabla L(\beta_t) = 0$
        - Where $\nabla^2L$ is the Hessian of L
    - For nonlinear least-squares, the Hessian has form: $\nabla^2L(\beta_t) = F'(\beta_t)^TF'(\beta_t) + \sum_{i = 1}^{n} (f(x_i; \beta_t) - y_i)\nabla^2f(x_i; \beta_t)$
    - The Gauss-Newton approximation neglects the second term and uses $\nabla^2L(\beta_t) \approx F'(\beta_t)^TF'(\beta_t)$
    - Using $\nabla L(\beta_t) = F'(\beta_t)^TF(\beta_t)$ we obtain the linear system:
        - $F'(\beta_t)^TF'(\beta_t)(\beta - \beta_t) + F'(\beta_t)^TF(\beta_t) = 0$
        - Solving this system yields the next iterate:
            - $\beta_{t+1} = \beta_t - (F'(\beta_t)^TF'(\beta_t)(\beta - \beta_t) + F'(\beta_t)^TF(\beta_t)) = 0$
    - For simplicity: $J_t = F'(\beta_t)$ and $F_t = F(\beta_t)$ and assume that $J_t^T J_t$ is invertible

- The Gauss-Newton Method:
    - Choose initial point sufficiently close to solution
    - For t = 0 to T, evaluate $F_t$ and $J_t$ then update the point
    - Details:
        - If the previous assumption of inversion is not possible, then use the pseudoinverse
        - Common stopping criterion is $\lVert J_t^T F_t \rVert_2 \leq tol$ which is exactly the norm $\lVert \nabla L(\beta_t) \rVert_2$
        - Method may converge to different stationary points depending on initial guess
        - Choosing initial point is crucial
    - Search Direction:
         - $\Delta \beta_t = -(J_t^T J_t)^{-1} J_t^T F_t$
         - Update can be written as the full-step iteration: $\beta_{t+1} = \beta_t + \Delta \beta_t$
         - If initial point is not close to solution, this step may fail to converge
         - Common remedy:
             - Damped Gauss-Newton step: $\beta_{t+1} = \beta_t + \alpha_t \Delta \beta_t$ where $\alpha_t \exists (0,1]$ where $\alpha_t$ is the learning rate
             - $\alpha_t$ can be chosen by backtracking line search:
                 - $\lVert F(B_t + \alpha_t \Delta \beta_t) \rVert_2^2 < \lVert F(\beta_t) \rVert_2^2$   