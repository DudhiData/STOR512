# Review of Mathematical Tools 

- Vectors and Vector Spaces
    - Mainly working with numbers in $\mathbb{R}$
    - Our data and parameters will be vectors of the form $x = (x_1, x_2, \dots, x_d)$, where $x_1, x_2, \dots, x_d \exists \mathbb{R}$
    - $x_i$ is the i-th element of the vector x, and d is the total # of elements in x (also known as the dimension of the vector)
    - The transpose operator $(.)^T$ transforms a column vector into a row vector
    - The set of all d-dimensional real-valued vectors is denoted by $\mathbb{R}^d$
    - The space $R^d$ is called a d-dimensional vector space (after equipping it with vector addition and scalar multiplication)
    - The zero vector is the vector whose entries are all 0
    - The all-ones vector is denoted by e:
        - $e^T = (1,1,\dots,1)$, which is understood as a column vector
     
- Vector Operations:
    - Can perform addition, subtraction, and scalar multiplication on vectors
    - Defined for any 2 vectors $x,y \exists \mathbb{R}^d$ and any scalar $\alpha \exists \mathbb{R}$

- Sparse Vectors:
    - Consider a high-dimensional vector $x \exists \mathbb{R}^d$, where d may range from tens of thousands to millions, or even larger
    - Vector x is said to be sparse if the number of its nonzero entries is much smaller than the number of zero entries
        - No universal threshold defining sparsity; depends on context and application
    - Sufficient to store only its nonzero entries
    - Often encoded using 2 arrays, where one stores indices of nonzero entries and the other stores the values of those entries

- Inner Product of 2 Vectors:
    - $<x, y> = x^T y = x_1 y_1 + x_2 y_2 + \dots + x_d y_d$
 
- Vector Norms:
    - Measures 'distance' from x to the origin in some sense
    - *$l_2$*-norm: $\lVert x \rVert_2 = \sqrt{x_1^2 + x_2^2 + \dots + x_d^2}$
    - *$l_1$*-norm: $|x_1| + |x_2| + \dots + |x_d|$
    - *$l_\infty$*-norm: $max({|x_1|, |x_2|, \dots, |x_d|})$
    - Properties:
        - $\lVert x \rVert_\infty <= \lVert x \rVert_2 <= \lVert x \rVert_1 <= \sqrt{d} \lVert x \rVert_2 <= d \lVert x \rVert_\infty$
        - For $x \exists \mathbb{R}^d$, the mapping $s(x) = \max_{u} {x^T u: \lVert u \rVert <= 1}$
        - Notions are symmetric
        - Inner product measures the correlation between 2 vectors, while the Euclidean norm measures their length
    - The angle between 2 vectors x and y is defined by:
        - $cos(x,y) = \frac{x^T y}{\lVert x \rVert_2 \lVert y \rVert_2}$
        - If the $\lVert x \rVert_2  = \lVert y \rVert_2 = 1$, then $cos(x,y) = x^T y$
        - If x is perpendicular to y, then cos(x,y) = 0; orthogonality
        - Since $-1 <= cos(x,y) <= 1$:
            -  $|x^T y| <= \lVert x \rVert_2 \lVert y \rVert_2$ (Cauchy-Schwarz Inequality)
         
- Norm Balls:
    - Given a norm $\lVert . \rVert$, we define a ball centered at $a$ of radius r as $\beta_r(a) := \{x \exists \mathbb{R}^d : \lVert x-a \rVert <= r\}$
    - For a given norm, the geometric shape of B(a) is different
    - We also denote B = $B_1(0)$, the unit ball centered at 0 of radius 1

- Orthogonal & Orthonormal Vectors
    - Two nonzero vectors $x,y \exists \mathbb{R}^d$ are orthogonal if $x^T y = 0$
    - Two nonzero vectors are orthonormal if:
        - $x^T y = 0$
        - $\lVert x \rVert_2 = \lVert y \rVert_2 = 1$
    - Any pair of orthogonal vectors can be normalized as dividing each entry of the vector by the Euclidean norm of the vector

- Orthogonal Systems:
    - A collection of vectors $L_k = \{v^{(1)}, v^{(2)}, \dots, v^{(k)}\}$, is an orthogonal if every pair of column vectors 