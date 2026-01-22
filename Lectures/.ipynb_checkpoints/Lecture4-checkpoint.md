# Lecture 4 (Linear Least-Squares and Applications)
- Review on Linear Regression:
    -  Given a dataset {($x_i$, $y_i$)} from i = 1 to n data points, where $x_i = (x_{i1}, \dots, x_{id})^T \exists R^d$, is a vector of inputs and $y_i \exists R$ is a scalar response
    -  A linear regression model assumes that the response is a linear function of the inputs plus some noise:
        - $y_i = B_0 + B_1 x_{i1} + \dots + B_d x_{id} + e_i = (x_i)^T B + e_i$
        - Where $x_i = (1, x_{i1},\dots,x_{id})$ is the augmented feature vector, and $B = (B_0, B_1, \dots, B_d)^T \exists R^{d+1}$ is the vector of regression coefficients
  
    -    Define the response vector, design matrix, and noise vector
    -    Then the linear regression model can be written in the compact matrix form $y = XB + e$
    -    Typically, the noise vector consists of random variables with 0 mean and a Gaussian distribution

- Optimization Model:
    - To estimate the coefficient vector B, we minimize the sum of squared errors, E is the squared Euclidean norm of the noise vector
    - Our problem becomes:
      
        $$
        \min_{B \exists R^{d+1}} {L(B) = \frac{1}{2} \sum_{i=1}^{n} (y_i - (x_i)^T B)^2 = \frac{1}{2} \lVert XB - y \rVert^2_2}
        $$
    - This is the linear least-squares problem, minimizing a sum of squared residuals (the 1/2 is for mathematical convenience canceling out the 2 when taking the derivative)

- Sparse Vectors:
    - Vector is said to be sparse if the number of its nonzero entries is much smaller than the number of zero entries

- Vector Norms:
    - The infinite norm is at most the Euclidean norm which is at most the l-1 norm which is at most the Euclidean norm times the square root of d which is at most d times the infinite norm
    - $cos(x,y) = \frac{x^Ty}{\lVert x \rVert_2 \lVert y \rVert_2}$
        - When cos(x,y) = 0, x is perpendicular to y, thus they are orthogonal
            - The absolute value of the numerator is at most the denominator (Cauchy-Schwarz)
        - Two nonzero vectors are orthonormal when the numerator is 0 and the vectors' Euclidean norms are 1
        - Any pair of orthogonal vectors can be normalized to get orthonormal vectors

- Orthogonal Systems:
    - A collection of vectors is an orthogonal system if every pair of vectors asides from themselves with themselves is orthogonal
    - Any OS is linearly independent
    - Any OS of d vectors in $R^d$ forms a basis of $R^d$
    - If V is a matrix whose columns are d orthonormal vectors, then $V^TV = I_d$, and hence $V^T = V^{-1}$
    - Any basis of $R^d$ can be converted into an orthornormal basis using Gram-Schmidt orthonormalization
 
- Simple Linear Regression (Example):
    - Consider a simple linear regression example with n = 3 data points and d = 1 input variable
    - In this case, the model is $y_i = B_0 + B_1 x_{i1} + e_i$, and the objective function becomes:

        $$
      L(B) = \frac{1}{2} [(B_0 + B_1 x_{11} - y_1)^2 + (B_0 + B_1 x_{21} - y_2)^2 + (B_0 + B_1 x_{31} - y_3)^2]
        $$
  
   - To compute the minimizer of L with respect to $\beta$, we apply Fermat's rule by setting the partial derivatives with respect to $\beta_1$ and $\beta_1$ equal to 0
   - This can be summarized into a system of two linear equations with the two unknowns $\beta_1$ and $\beta_1$ can be written compactly as the normal equations $X^T X \beta = X^T y$
       - Where design matrix X has a column of 1's and a column of the $x_{i1}$ and the response vector y is just $y_i$
       - We can assume that $X^TX$ is invertible (requires the columns of X to be linearly independent)
           - Thus, $\beta = (X^TX)^{-1}X^Ty$
               - We can apply Fermat's rule to the unconstrained problem $\min_{\beta} L(B)$ by setting $\nabla_\beta L(\beta) = 0$
               -   $\nabla_\beta L(\beta) = X^T(X\beta - y)$
               -   Therefore, the optimality condition is equivalent to $X^TX\beta = X^Ty$
   - If $X^TX$ is invertible then $\beta_* = (X^TX)^{-1}X^Ty$
       -  This occurs when X has full column rank (its d+1 columns are linearly independent)
           - If n > d + 1 and the columns of X are not collinear, then $X^TX$ is invertible
       - Since $X^TX$ is symmetric positive definite, we can use Cholesky factorization $X^TX = LL^T$, where L is a lower triangular and invertible matrix
       - Consequently, $(X^TX)^{-1} = L^{-T}L^{-1}$ (4)
   - If $X^TX$ is not invertible, then we can't use (4) since it doesn't have a unique solution
       - Since $X^TX$ is symmetric PSD, we can consider its eigendecomposition $X^TX = V\lambda V^T$, where V has the orthonormal eigenvectors of X as columns and $\lambda = diag(\lambda_1, \dots, \lambda_r, 0, \dots, 0)$, with $\lambda_i$ > 0 being the r positive eigenvalues
       - The Moore-Penrose pseudoinverse of $X^TX$ is then $(X^TX)^+ = V\lambda^+V^T$
       - A minimum-norm solution is then $B_* = (X^TX)^+X^Ty = X^+y$, where $X^+$ denotes the psuedoinverse of X
       - When both n and d are large, computing $X^TX$ and its inverse or pseudoinverse become computationally expensive


- SVD Decomposition Solution:
    - We can use SVD to compute the pseudoinverse and then obtain the solution $B_* = X^+y$
    - To compute $X^+$, we first compute the SVD of X:
        - $X = U\sum V^T$, where U and V are orthonormal matrices and $\sum$ is a diagonal matrix containing the singular values of X
    - We then form the diagonal matrix $\sum^+$ by taking the reciprocal of the nonzero singular values and leaving 0s unchanged and compute $X^+ = V\sum^+U^T$
        - In Python, use numpy's linalg module to do linalg.pinv(X) to get $X^+$ and then do X_plus.dot(y) to get the best beta  