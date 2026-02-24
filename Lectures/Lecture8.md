# Lecture 8 (Classification)

- Predicting categorical responses --> classification
- Also estimating class membership probabilities
- Common methods include: kNN, logistic regression, linear discriminant analysis, quadratic discriminant analysis, and naive Bayes
- Another important classification method is the support vector machine (originally for binary classification)


- Binary Classification:
    - Exactly two possible classes
    - Value of response variable is called class label
    - Common choice of labels is {-1, 1}, {1, 0}
    - Logistic Regression:
        - Widely used model for binary classification
        - Models probability that an observation belongs to a given class using the logistic function
        - Main goal is to estimate the model parameters that best explain observed class labels
        - Let $y \exists {0, 1}$ denote the class label and $x \exists \mathbb{R}^d$ as the feature vector
        - The conditional probability of the positive class is modeled by: $p(x) = \mathbb{P}(y = 1 | x) = \frac{exp(\mu + x^T \beta)}{1 + exp(\mu + x^T \beta)}$
            - Since there are only two classes, $\mathbb{P}(y = 0 | x) = 1 - p(x)$
            - The scalar $\mu$ (intercept) and the vector $\beta$ (coefficients) are unknown parameters to be estimated from the data
            - To estimate $(\mu, \beta)$, we apply the maximum log-likelihood principle:
                - The likelihood function is $l(\mu, \beta), \prod_{i: y_i = 1} p(x_i) \prod_{i: y_i = 0} (1 - p(x_i))$
                - Taking logarithms and using the equation above, we obtain:
              
$$
log(l(\mu, \beta)) = \sum_{i: y_i = 1} log(p(x_i)) + \sum_{i: y_i = 0} log(1 - p(x_i)) = \sum_{i: y_i = 1} [\mu + x_i^T \beta - log(1 + exp(\mu + x_i^T \beta))] - \sum_{i: y_i = 0} log(1 + exp(\mu + x_i^T \beta)) 
$$

- This ultimately ends up being: 

$$
-\sum_{i = 1}^{n} [log(1 + exp(\mu + x_i^T \beta)) - y_i(\mu + x_i^T \beta)]
$$

   - Maximizing the log-likehood is equivalent to minimizing the logistic loss:
       - $\min_{(\mu, \beta) \exists \mathbb{R}^{d+1}} L(\mu, \beta) = \sum_{i = 1}^{n} (log(1 + exp(\mu + x_i^T \beta)) - y_i(\mu + x_i^T /beta))$
       - The key properties of L:
           - Each term is the sum of a convex and linear function
           - Since $l(t) = log(1 + e^t)$ is convex and composition with an affine map preserves convexity, each $L_i$ is convex
           -  Therefore, L is convex, and any point $\mu_*, \beta_*)$ satisfying $\nabla_{(\mu, \beta)}L(\mu_*, \beta_*) = 0$ is a global minimizer of the above problem

- Solution Methods for Logistic Regression:
    - The convex optimization problem can be solved by first-order methods such as (accelerated) gradient methods, stochastic gradient methods, and randomized coordinate descent
    - Accelerated Gradient:
        - Since L is convex and $L(\mu, \beta)$ is smooth, we can apply Nesterov's accelerated gradient method
        - Let $w = (\mu, \beta^T)^T \exists \mathbb{R}^{d+1}$ be the parameter vector
        - Define the data matrix $X \exists \mathbb{R}^{n x (d+1)}$ whose i-th row is $X_i^T = (1, x_i^T)$
        - Then the related optimization problem can be written as $L(w) = \sum_{i = 1}^{n} [log(1 + exp(X_i^T w)) - y_i X_i^T w]$
        - The gradient of L is $\nabla L(w) = \sum_{i = 1}^{n} (p_i(w) - y_i) X_i = X^T (p(w) - y)$
            - Where $p_i(w) = \frac{exp(X_i^T w)}{1 + exp(X_i^T w)}$ and $p(w) = (p_1(w), \dots, p_n(w))^T$
            - The gradient is a Lipschitz continuous with constant $L = \frac{1}{4} \lambda_{max} (X^TX)$
        - Summarized below:
            - Choose $w_0 \exists \mathbb{R}^{d+1}$ and tolerance
            - Set $\hat{w_0} = w_0, \tau_0 = 1$ and compute L
            - Step 1: $w_{t+1} = \hat{w_t} - \frac{1}{L} \nabla L(\hat{w_t})$
            - Step 2: If $\lVert w_{t+1} - w_t \rVert \leq tol$, terminate
            - Step 3: Update $\tau_{t+1} = \frac{1}{2} (1 + \sqrt{1 + 4 \tau_t^2})$ & $w_{t+1} + \frac{\tau_t - 1}{\tau_{t+1}}(w_{t+1} - w_t)$
        - Does not guarantee monotonic decrease of $L(w_t)$ but converges faster in theory

- Iteratively Reweighted Least Squares Method:
    - By Fermat's rule, an optimal solution $w_*$ satisfies $\nabla L(w_*) = 0$
    - Linearizing this nonlinear equation around the current iterate $w_t$ via a first-order Taylor Expansion gives $0 = \nabla L(w) \approx \nabla L(w_t) + \nabla^2 L(w_t)(w-w_t)$
    - Solving the resulting linear system yields Newton update: $w_{t+1} = w_t - \nabla^2 L(w_t)^{-1} \nabla L(w_t)$
    - The Hessian of L is: $\nabla^2 L(w) = X^T P(w) X$ where $P(w) = diag(p_1(w)(1-p_1(w)), \dots, p_n(w)(1-p_n(w)))$
    - The gradient is $\nabla L(w) = X^T (p(w) - y)$ where $p(w) = (p_1(w), \dots, p_n(w))^T$
    - Substituting into the update: $w_{t+1} = (X^T P_t X)^{-1} X^T P_t[X w_t + P_t^{-1}(y-p_t)]$, where $P_t = P(w_t)$ and $p_t = p(w_t)$
    - Define the adjusted response $z_t = X w_t + P_t^{-1}(y-p_t)$
    - The $w_{t+1}$ step solves the weighted linear least squares problem: $w_{t+1} = \min_{w} \frac{1}{2} \lVert Xw - z_t \rVert_{P_t}^2$ where $\lVert u \rVert_P^2 = u^T P u$
    - Algorithm:
        - Choose $w_0 \exists \mathbb{R}^{d+1}$ and tolerance
        - For t = 0,...,T:
            - Compute $p_t = p(w_t), P_t = diag(p_t(1-p_t))$
            - If $\lVert \nabla L(w_t) \rVert \leq tol$ terminate
            - Update $z_t = Xw_t + P_t^{-1} (y-p_t)$ and $w_{t+1} = (X^T P_t X)^{-1} X^T P_t z_t$
    - Common initialization point is $w_0 = 0$
    - May overshoot when point is far from optimal --> use damped Newton step where we just multiply the Hessian * gradient term when updating $w_{t+1}$ by a constant between 0 and 1 chosen by a line search

- Support Vector Machines:
    - Given a training dataset ${(X^{(i)}, y_i)}_{i = 1}^n$ where $X^i \exists \mathbb{R}^d$ is the feature vector and $y_i \exists \{-1, 1\}$ is the class label
    - Our goal is to find the separating hyperplane $H = \{X \exists \mathbb{R}^d | w^T X + \mu = 0\}$ where $w \exists \mathbb{R}^d$ is the normal vector and $\mu \exists \mathbb{R}$ is the intercept
    - The hyperplane should separate the two classes as best as possible
    - If the data is linearly separable, there exist infinitely many separating hyperplanes
    - SVM selects the one with the maximum margin (largest distance to the closest data points)
    - Consider 2 hyperplanes parallel to H: $w^T X + \mu = 1$ and $w^T X + \mu = -1$
    - These constraints can be written as: $y_i(w^T X^i + \mu) \geq 1$ for i = 1 to n
    - The distance between these 2 hyperplanes is $\frac{2}{\lVert w \rVert_2}$ --> therefore, maximizing the margin is equivalent to minimizing $\lVert w \rVert_2$
    - The optimization problem is the convex quadratic program:
        - $\min_{w \exists \mathbb{R}^d, \mu \exists \mathbb{R}} \frac{1}{2} \lVert w \rVert_2^2$ subject to $y_i(w^T X^i + \mu) \geq 1$ for i = 1 to n
        - The objective enforces a large margin
        - The formulation assume perfect linear separability
        - Constraints ensure correct classification of all training points
        - Known as hard-margin SVM
    - Soft-Margin SVM:
        - Real-world datasets are rarely perfectly separable
        - We introduce slack variables $n_i \geq 0$ to allow misclassification
        - The parameter $C > 0$ controls the trade-off between margin size and classification error
        - $\min_{w, \mu, n \geq 0} \frac{1}{2} \lVert w \rVert_2^2 + C * \sum_{i = 1}^{n} n_i$ subject to $y_i(w^T X^i + \mu) \geq 1- n_i$ for i from 1 to n

