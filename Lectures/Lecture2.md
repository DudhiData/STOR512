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

   - Since $s(t) = B_1 t +  B_2 \log(t)$, we can write this as a least squares problem:

$$
\min_{B_1, B_2} (0.5 * \sum_{i=1}^{n} (B_1 t_i +  B_2 \log(t_i) - \hat{s}(t_i))^2
$$

   - Consider a matrix X with values of t as one column and values of log(t) as another column, and a vector y as our output
   - Then we can change our above problem to:

$$
\min_{B \exists R^2} (0.5 * (\lVert XB - y \rVert)^2) = 0.5 * \sum_{i=1}^{n} (X_i^T B - y_i)^2)
$$

   - We can solve this and obtain a unique solution:
     
$$
B^* = (X^T X)^{-1} X^T y
$$

   - Comparing these values and the actual data, we can see there are small errors due to noise
   - We can also use the maximum likelihood principle:
      - Assuming the underlying model is linear and given by $s(t) = B_1 t + B_2 log(t) + e$, where e is a Gaussian noise with 0 mean and variance $o^2$
      - From this model, we have $e = s(t) - B_1 t - B_2 log(t)$, which is a Gaussian random variable
         - Our observed data provides i.i.d. samples of e, given by $e_i = \hat{s_i} - B_1 t_i - B_2 log(t_i)$
         - The joint probability density of $(e_1, \dots, e_n)$ can thus be derived using the multiplied probabilities derived from the PDF of the normal distribution and our estimators
            - We maximize the log of this joint density formula with respect to $B_1$ and $B_2$, which eventually boils down the exact least-squares problem derived earlier
          
 - Example 3: Markowitz' Portfolio Selection
    - Assume that we have d assets over a given investment period. Let $x_i$ denote the amount of capital invested in asset i, measured in dollars at the price at the beginning of the period
        - Collecting all assets, we obtain the portfolio vector x
    - Let $p_i$ denote the relative price change of asset i over the period, defined as the ratio of the price change to the initial price
        - The overall return of the portfolio x is given by the $r = p^T x$
        - The optimization variable is the portfolio vector x
        - The constraints include a total investment budget B, $\sum_{i=1}^{n} x_i = B$ and nonnegativity constraints
        - Since asset prices are uncertain, the vector p is modeled as a random vector with known mean m and a covariance matrix A
        - With a fixed portfolio x, the return r is therefore a random scalar variable with mean $m^T x$ and variance $x^T Ax$
    - The classical selection model seeks to minimize the return variance while satisfying the budget constraint, a minimum expected return requirement, and upper bounds on individual asset holdings
    - This is a constrained convex quadratic program

 - Example 4: Support Vector Machines
    - Commonly used for binary classification tasks, where some examples labeled positive and others negative
    - Motivating Example:
       - We are given 10 data points in $R^2$, denoted by D = {($x^i, y_i$)} from i = 1 to 10, and where $x^i$ = ($(x_1)^i, (x_2)^i) \exists R^2$ and $y_i \exists$ {-1, 1} is the label
       - Our goal is to find the hyperplane H that separates the data points with positive labels $y_i$ = 1 from those with negative labels $y_i$ = -1
       - The hyperplane H is defined by $w_1 x_1 + w_2 x_2 + b = 0$, where $w_1$, $w_2$, and b are unknown coefficients
       - To avoid the situation where some data points lie exactly on this hyperplane, we introduce 2 parallel hyperplanes that form a band around the hyperplane H:
          - $H_-: w_1 x_1 + w_2 x_2 + b = -1$
          - $H_+: w_1 x_ 1 + w_2 x_2 + b = 1$
       - Note that the RHS -1 and 1 can be replaced by an values -c and c respectively
       - However, since we can scale the coefficients by dividing them by c to recover the RHS -1 and 1, we choose c = 1 without loss of generality
       - To strictly separate the data points, we aim to maximize the width of this band, which is exactly the difference between the two hyperplanes $H_-$ and $H_+$
       - The distance from a given point $a \exists R^2$ to a plane $H_+$: $w_1 x_1 + w_2 x_2 + b = 1$ is $|w_1 a_1 + w_2 a_2 + b - 1| / ((w_1)^2 + (w_2)^2)^{1/2}$
       - If a lies on $H_-$, then this distance equals the distance between the two planes
       - Our goal is to maximize this distance: max $2/((w_1)^2 + (w_2)^2)^{1/2}$, or min $0.5((w_1)^2 + (w_2)^2)$
       - The constraints are $w_1 (x_1)^{i} + w_2 (x_2)^{i} + b <= -1$ for negative labels and >= 1 for positive labels
       - Hence our final model can be simplified to:

$min_{w_1, w_2 \exists R}$:

$$
0.5 ((w_1)^2 + (w_2)^2) \text{ subject to } y_i (w_1 (x_1)^{i} + w_2 (x_2)^{i} + b) >= 1
$$
    
   - This is known as the SVM model

 - Generalization to Multi-Dimensional Settings:
    - We are given a set of data points $\({(x^{i}, y_i)}_{i=1}\)^{m}$, where x $\exists R^d$ and the label $y_i \exists$ {-1,1}
    - Assume that we draw $x^i$ in the $R^d$ space and want to find a hyperplane H: $w^T x + b = 0$  that separates all the points associated with a label $y_i$ = 1 into one half-space and the others into another half-space, and w $\exists R^d$ is its normal vector, and b $\exists R$ is an intercept
    - Problem Formulation:
        - We assume there exist 2 hyperplanes $w^T x + b = 1$ and $w^T x + b = -1$, which are parallel to the separating hyperplane H and separate the dataset into 2 half-spaces
        - The distance between these two parallel hyperplanes is $2/(\lVert w \rVert)_2$,
        - Maximizing the margin is equivalent to minimizing the Euclidean norm of w, or minimizing $\((\lVert w \rVert)^2\)_2$
        - We can formulate this SVM model into the optimization problem:
            - $min_{w \exists R^d}$ $0.5 \((\lVert w \rVert)^2\)_2$ subject to $y_i (w^T x^{i} + b) >= 1$ which is a convex quadratic problem

 - Example 4: Handwritten Digit Recognition Example
     - We are given a collection of handwritten digit images taken from ZIP codes on US postal envelopes
     - Each image corresponds to a single digit extracted from 5-digit ZIP code
     - The images are represented as 16x16 grayscale arrays with 8-bit intensity values, where each pixel intensity ranges from 0-255
     - As a preprocessing step, these images are normalized to have approximately the same size and orientation
     - We want to predict the identity of each image (0..9) using the matrix of pixel intensities

- Machine Learning:
    - Study of computer algorithms that can automatically improve through experience and the use of data
    - Alternatively, ML can be broadly defined as computational methods using experience to improve performance or make accurate predictions
    - 3 Learning Scenarios:
        - Supervised Learning:
            - The learner is provided with a set of labeled examples as training data to learn a machine learning model, which is then used to make predictions on unseen data points
            - Most common learning paradigm -> closely associated with classification, regression, and ranking
        - Unsupervised Learning:
            - Learner is given unlabeled training dataset and attempts to discover patterns or structures without explicit output labels
            - Often difficult to quantitatively evaluate the performance of the learned model
            - Aims to ID hidden structures in data, such as clusters, groups, or low-dimensional representations (e.g. PCA)
        - Reinforcement Learning:
            - Learner actively interacts with an environment and may affect it by taking actions
            - After each action, the learner receives an immediate reward, and the overall goal is to maximize the cumulative reward over time
            - Training and testing phases intertwined
            - Examples: games against humans
    - Common Classical ML Tasks:
        - Aims to build model based on mathematical tools and sample data without being explicitly programmed to do so
        - Classification:
            - Assigning discrete category to each item
        - Regression:
            - Focuses on predicting a continuous real-valued output for each item
        - Clustering:
            - Task aims to partition a set of items into homogeneous groups based on similarity (often for large-scale datasets without labels)
        - Other tasks include: ranking, dimensionality reduction, and manifold learning
    - Key Terminologies and Concepts:
        - Examples: individual instances of data used in ML task
        - Features: set of attributes associated with each example (represented as vectors)
        - Labels: values or categories assigned to examples
        - Hyperparameters: parameters not learned by training algorithm but specified by user as inputs to algorithm (e.g. learning rate and regularization parameters in least-squares or LASSO)
        - Training sample: set of examples used to train/fit a ML mdel
        - Validation sample: set of examples (separate) used to tune hyperparameters and select appropriate model settings
        - Test sample: set of examples used only to evaluate final performance of trained model
            - Performance typically measured using training loss, test loss, or classification error
        - Loss function: function that measures discrepancy between predicted output and true label (usually objective function in the underlying optimization problem)
        - Hypothesis set (model class): collection of candidate functions that map features to labels or responses, from which the learning algorithm selects a model
    - Generalization, Underfitting, Overfitting:
        - A model generalizes well if it can make accurate predictions on unseen data
        - In supervised learning, a model is trained on a finite training dataset using a chosen hypothesis set  -> as a result prediction errors on new data are unavoidable
        - Our goal is to minimize the prediction error on unseen data
        - Fundamental trade-off between training error and test error, as well as between model complexity and sample size
        - Overfitting: when model is too complex relative to the sample size, the result is overfitting
        - Underfitting: when model is too simple to capture the underlying structure, the result is underfitting   




