# Lecture 3
- In ML, we train models using a finite training dataset
    - The true objective is not to perform well on training data but to do well on unseen test data
    - Generalization refers to the ability of a learned model to make accurate predictions on new data
    - A model with very low training error may still perform badly on new data -> overfitting
    - An overly simple model may fail to capture important patterns -> underfitting
    - Therefore ML requires a trade-off between:
        - Model complexity
        - Training accuracy
        - Sample size
    - Key Takeaway:
        - Good optimization != Good generalization
        - 
- Overfitting in Machine Learning:
    - Occurs when a model is too complex and fits noise rather than true data-generating process
    - Achieves very low training error but poor test performance
    - Common causes:
        - Overly rich hypothesis class
        - Small training datasets
        - Insufficient regularization
    - Associated with low bias and high variance
    - Fixes: regularization, early stopping, CV, more data
     
- Underfitting in Machine Learning:
    - Occurs when chosen model is too simple to capture underlying structure of data
    - Will perform poorly on training and test data
    - Often results from:
        - Overly restrictive hypothesis class (e.g. linear model for nonlinear data)
        - Excessive regularization
        - Insufficient training time
    - Associated with high bias and low variance
    - Increase model complexity, reduce regularization, or add informative features to combat this

- Optimization Models in ML:
    - Unlike operations research, optimization in ML or DS is only 1 core step in the overall pipeline
    - We describe a unified optimization template for supervised learning:
        - Given a finite set of training examples: {($x_i$, $y_i$)} from i = 1 to n, where $x_i$ is a feature vector and $y_i$ is a response or label
        - Our goal is to learn a prediction function that maps feature vectors to outcomes
    - To model this, we often parameterize the learner as a function U(x; B) where B denotes model parameters (e.g. linear models take the form U(x; B) = $x_T$ B
    - Let l be a loss function that measures discrepancy between predicted outcome $\hat{y}_i = U(x_i ;B$  and the true outcome $y_i$
    - The empirical/average loss over the training set is defined as L(B; {$x_i$ , $y_i$}) := $\frac{1}{n} \sum_{i=1}^{n} l(U(x_i ;B), y_i)$
    - The learning objective is to minimize the empirical loss over a prescribed hypothesis set F
    - Leading to the following optimization problem:

    $$
    \min_{B \exists F} {L(B; {x_i, y_i}):= \frac{1}{n} \sum_{i=1}^{n} l(U(x_i;B), y_i)}
    $$
  
    - Without further structure on the model U and the loss function l, this problem is challenging
    - So we must specify appropriate models and loss functions
    - The training sample size n also plays a huge role, affecting generalization performance and choice of optimization algorithm

- ML and Optimization:
    - Primary goal: Generalization
    - Data-driven: High
    - Model certainty: Low
    - Problem scale: Very large
    - Exact solution: Rarely needed

- Examples:
    - Example 4:
        -  Assume that our model is linear as U(x:B) = $B_0 + \sum_{i=1}^{n} B_j x_{ij}$ and our loss is just a squared loss, $l(s,t) = \frac{1}{2} (s-t)^2$
        -  Then we can write our optimization problem as:
      
        $$
        \min_{B \exists R^{d+1}} \frac{1}{2n} \sum_{i=1}^{n} (B_0 + \sum_{j=1}^{d} x_{ij} B_j - y_i)^2     
        $$
    
        - This is exactly a linear regression problem
  
    - Example 5:
        - If we replace least-squares loss by a logistic loss $l_2(s,t) = log(1+exp(s)) - ts$, then we obtain:
          
          $$
          \min_{B \exists R^{d+1}} \frac{1}{n} \sum_{i=1}^{n} [log(1 + exp(B_0 + \sum_{j=1}^{d} x_{ij} B_j)) - y_i(B_0 + \sum_{j=1}^{d} x_{ij} B_j)]
          $$
        - Which is a logistic regression problem in binary classification

- Solving Optimization Problems in Practice:
    - Exact solutions to optimization problems are rare in practice
    - Only few special cases admit closed-form solutions such linear least-sqaures or leading eigenvalue of covariance matrix
    - We usually get approx. solutions
    - Multiple stages: modeling, preprocessing, solution methods, and post-processing

- Stage 1: Mathematical Modeling:
    - Process of translating a real-world problem described in words and data into mathematical formulation using functions, equations, and constraints
    - ID 3 core components:
        - Decision variables
        - Objective function
        - Constraints

- Stage 2: Preprocessing
    - Simplify model by removing redundant variables or constraints
    - Collect and preprocess data: handle missing values, normalize features, or transform variables as needed
    - Classify optimization problem into known class (e.g. if all functions are linear, the problem is a linear program -> use simplex)
    - Analyze feasibility and optimality: determine whether feasible and optimal solutions exist, and whether they are unique

- Stage 3: Solution Methods:
    - Level 1:
        - Use existing software tools such as Excel, R, MATLAB, SAS, SciPy, Scikit-Learn, or TensorFlow
    - Level 2:
        - Implement known algorithms in programming languages such as Python, Julia, C (e.g. implementing stochastic gradient descent to train NNs)
    - Level 3:
        - Customize or adapt existing algorithms to specific applications
    - Level 4:
        - Develop new optimization algorithms
     
- Stage 4: Post-processing
    - Solution quality is evaluated and validated
    

    
