# Lecture 5

- Eigenvalue decomposition:
    - Import numpy, then use np.linalg.eig(matrix) to get the eigenvalues and eigenvectors
    - If all eigenvalues are $\gt$  0, then positive definite
    - If all eigenvalues are $\ge$ 0, then positive semi-definite

- Convexity:
    - $f(x) + f^{\prime}(x)(y-x) \le f(y), \forall x,y \exists \mathbb{R}$
    - A function f is convex if the tangent line to the graph of f at any point always likes below the graph of f
    - Second derivative is non-negative (Hessian is positive semi-definite)
    -  Take any 2 points x, y, and let z be any point in between them
        - $z = \theta x + (1-\theta)y$ where $\theta$ exists in a closed interval between 0 and 1
        - A function f is convex if $f(\theta x + (1-\theta)y \le \theta f(x) + (1-\theta) f(y)$,  $\forall x,y, \theta \exists [0,1]$
        - Called Jensen Inequality

- Convexity for Multivariate Functions:
    - A function f: $\mathbb{R}^d$ -> $\mathbb{R}$ is called convex if for any $x, y \exists \mathbb{R}^d$ and any $\alpha \exists [0,1]$:
        - $f((1-\alpha)x + \alpha y) \le (1-\alpha) f(x) + \alpha f(y)$
        - If differentiable, we can characterize it as:
            - $f(x) + \nablaf(x)^T (y-x) \le f(y), \forall x,y \exists \mathbb{R}^d$
