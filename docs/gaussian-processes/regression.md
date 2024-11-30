# Regression

In general, the main goal of regression is to go from some training data $\mathcal{D}$ to a function that makes predictions for all possible input values. To do this, we must make assumptions about the underlying function that generated that data. We could do so by, say, assuming a particular functional form like that it is a linear map, and then finding the linear map which best fits the data. Or, we could encode our beliefs about the function into a prior probability distribution, i.e. that these are smooth or periodic functions. This is a much more general class of functions to choose from and thus a much weaker assumption, so ideally this is a more attractive way of doing inference. However, given that we are supposed to then select a particular function from this uncountably infinite space of functions, it seems completely infeasible to actually do this. This is where Gaussian processes come to the rescue, allowing us to describe our beliefs about a space of functions in a particularly computationally efficient way. 

There are several ways to interpret a Gaussian process regression model. One can think of a Gaussian process as defining a distribution over functions, and inference taking place directly in the space of functions. This is the so-called *function space view*. One can also think about it in an equivalent *weight space view*, which we'll now describe. We will then connect this perspective to the function space view.


## The Weight Space View

It will be simplest to start with a review of how standard linear regression fits into the Bayesian framework. 

!!! danger "Conventions"
    Rasmussen and Williams define the training set $\mathcal{D}$ of $n$ observations, $\mathcal{D}((\mb{x}_i, y_i)\,|\, i = 1, \ldots, n)$, where the input vectors $\mb{x}$ have dimension $d$ (or $D$), and they aggregate the inputs (or covariates) into a *design matrix* $X$ with dimension $d \times n$. This is opposite to how `torch` tends to handle these things, where it assumes that rows correspond to samples and columns correspond to index. As annoying as it is, to translate into `torch` code it will be convenient to think of $X$ as a $n\times d$ matrix.

    I find all of this matrix notation generally confusing because it is too easy to mix up the fact that there are two completely ontologically separate spaces. Maybe that's just the quantum mechanic in me, where I like to think of everything as happening in the same Hilbert space. However, it also obscures the fact that parameter space and sample space can (and do) have completely different geometries. So, instead of following directly along with Rasmussen and Williams, it will be useful to extend their notation in a way that more directly exposes this geometry. After all, we should work with a notation that naturally meshes with the structure of the objects we are working with.

    I will use $w^a$ to denote coordinates on parameter space, with $a,b, \ldots = 1, \cdots, d$. It's useful to think of this as an upper index because an infinitesimal $\ud w^a$ should definitely exist in the tangent space. I'll use $i, j, \ldots = 1, \ldots, n$ to denote indices in *sample space*, i.e. to label samples. We will assume that the samples are i.i.d. and that the noise is homoskedastic, so that the local geometry of sample space is just plain Euclidean. Topologically, sample space is $\mathbb{R}^n/\lab{S}_n$, where $\lab{S}_n$ is the symmetric permutation group, since we don't care what order the samples come in. This actually doesn't seem to be studied that much, except for this recent paper [[2010.08039]](https://arxiv.org/abs/2010.08039), and it's not necessarily clear to me whether one gains a lot by considering that orbifold. To that end, I'll not worry too much about that metric on sample space and instead freely raise and lower sample indices. Of course, repeated indices indicate an implicit summation.

    Honestly, this makes it a lot easier to see what's going on, especially when there are multiple unrelated spaces floating around.

Let us consider the linear map $f(x) = x_a w^a$, which we will assume is related to the noisily observed value $y$ by $y = f(x) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma_n^2)$ a normally distributed variable with standard deviation $\sigma_n$. We will stick to the case where $x, y  \in \mathbb{R}$ and write $x_a = (1, x)$ so that a general linear function $f(x) = w_0 + w_1 x$ can be written as an inner product. For $n$ samples, we can just append an index on everything and write $y^i = X^i_a w^a + \varepsilon^i$ (a different realization of $\varepsilon$ for each sample) and collect all of the noisy measurements into $y^i = \mb{y}$.

We will assume that the **true model** (without noise, or said differently the mean of the true data generating process) has $w^a = (1, 1)$. This is the true model if we assume that $\sigma_n$ represents some additional, obstructing noise that is a side effect of the measurement process; or we can simply think of $f(\mb{x})$ as a model for the mean of the distribution. We can construct the generative model for this data by assuming that each point is pulled from a normal distribution (it is) with some $\sigma_n$ standard deviation and mean centered around $f(x^i) = x^i_a w^a$ for some parameters $w^a$. As an example, we can plot data generated by this random process, compared to model $f(\mb{x})$:

=== "Plot"
    ```plotly
        {"file_path": "./gaussian-processes/code/regression/line_data.json"}
    ```
=== "Code"
    ```py title="Generative Model"
        --8<-- "./docs/gaussian-processes/code/regression/regression.py:line_data"
    ```

The **likelihood** of the data is then 

$$
\begin{aligned} p(\mb{y} | X, \mb{w}) &= \prod_{i = 1}^{n} p(y_i | \mb{x}_i , \mb{w}) = \prod_{i = 1}^{n} \frac{1}{\sqrt{2 \pi \sigma_n^2}} \exp\left(-\frac{(y_i - \mb{x}_i \cdot \mb{w})^2}{2 \sigma_n^2}\right) \\
&= \frac{1}{(2 \pi \sigma_n^2)^{n/2}} \exp\left(-\frac{1}{2 \sigma_n^2}|\mb{y} - X \mb{w}|^2\right) 
\end{aligned}
$$

Here, we have introduced the design matrix $X = X_i^a$, so $X \mb{w} = X_i^a w_a$. Generally in the Bayesian formalism we should also specify a prior on the weights $\mb{w}$. We will assume that $\mb{w} \sim \mathcal{N}(0, \Sigma_p) \propto \exp(-\frac{1}{2} \mb{w}^\top \Sigma_p^{-1} \mb{w})$. It seems like it will actually be convenient to use this prior covariance as a metric; after all, it is the Fisher information metric for the parameters $w_a$ 


We can then construct a posterior distribution for the weights via the product rule, namely

$$ p(\mb{w} | \mb{y}, X) = \frac{p(\mb{y} | X, \mb{w}) p(\mb{w})}{p(\mb{y} | X)}\,. $$

The bottom doesn't depend on $\mb{w}$, so we can just focus on the top and then normalize. The numerator has the form

$$ p(\mb{y} | X, \mb{w}) p(\mb{w}) \propto \exp\left(-\frac{1}{2 \sigma_n^2}|\mb{y} - X \mb{w}|^2 - \frac{1}{2} \mb{w}^\top \Sigma_p^{-1} \mb{w}\right)$$

The easiest way to group terms is to just take a derivative of the (negative) argument with respect to $\mb{w}$, yielding

$$ \sigma_n^{-2} X^\top (X \mb{w} - \mb{y}) + \Sigma_p^{-1} \mb{w} = 0$$

which has the straightforward solution

$$ \bar{\mb{w}} = \sigma_n^{-2} \left(\Sigma_p^{-1} + \sigma_n^{-2} X^\top X\right)^{-1} X^\top \mb{y},$$

while the concavity is given by

$$ A \equiv \frac{1}{\sigma_n^2} X^\top X + \Sigma_p^{-1}\,.$$

This means that the posterior distribution on the weights is

$$ p(\mb{w} | \mb{y}, X) = \mathcal{N}(\bar{\mb{w}}, A^{-1})\,.$$

We can very easily plot and sample this posterior, for $\sigma = 0.1$ and $\sigma_n = 0.1$.

=== "Plot"
    ```plotly
        {"file_path" : "./gps/code/regression/posterior_sampling.json"}
    ```

=== "Code"
    ```py title="Posterior Sampling"
        --8<-- "./docs/gps/code/regression/regression.py:posterior_sampling"
    ```

If we instead assume that the data is noiseless, then this collapses the posterior to a much smaller region. Below, we take $\sigma_n = 0.01$; note the much smaller range of the contour plot, and much tighter grouping of the randomly sampled lines.

=== "Plot"
    ```plotly
        {"file_path" : "./gps/code/regression/posterior_sampling2.json"}
    ```

=== "Code"
    ```py title="Posterior Sampling"
        --8<-- "./docs/gps/code/regression/regression.py:posterior_sampling2"
    ```

What we are left with is a posterior distribution $p(\mb{w} | \mb{y}, X)$. We can convert this into a straight-up Bayesian prediction for a new point $\mb{x}_*$, given the data $X$, by writing

$$
    p(f_* | \mb{x}_*, \mb{y}, X) = \int\!\ud\mb{w}\, p(f_* | \mb{x}_*, \mb{w}) p(\mb{w} | \mb{y}, X)\,,
$$

where we've taken the same generative model (that depends on the parameters $\mb{w}$) and just integrated it against the generative model. This just works out to be

$$
    p(f_* | \mb{x}_*, \mb{y}, X) = \mathcal{N}(\sigma_n^{-2} \mb{x}_* A^{-1} X^\top \mb{y}, \mb{x}_*^\top A^{-1} \mb{x}_*)
$$


## Projections of Inputs into Feature Space

Now, linear regression is great but we would like to go beyond to nonlinear regression, increasing the expressivity of our models. An easy way to do this is, instead of restricting ourselves to models of the form $f(x) = w_0 + w_1 x$, we can just add another basis vector[^1] and instead consider models of the form $f(x) = w_0 + w_1 x + w_2 x^2 + \cdots$.

[^1]: Literally, another basis vector in the space of functions.

Defining the $d$-dimensional "feature" vector $\Phi^a(x) = (1, x, x^2, \cdots)$ and extending the vector of parameters $\mb{w}$ to have length $d$ so that $f(x)$ can express an arbitrary degree-$d$ polynomial, we can write an arbitrary model in this class as 

$$ f(x) = \phi^a(x) w_a\,, $$

and we'll denote the aggregation of the feature vectors of all samples $\phi^a(x_i)$ as $\Phi = \Phi(X) \equiv \Phi^a_i$. Generally, to conform with `torch`'s conventions we will think of $\Phi$ as a $n \times d$ matrix. We'll write $\Phi(X) \mb{w} = \Phi_i^a w_a$ so that $w_a$ appears as a column vector, if we are doing straightup matrix multiplication.


Note that since we will consider $\phi(\mb{x})$ to be a **row vector**, then $\Phi(X)$ is a $n \times d$ matrix, while $\mb{w}$ is a $d$-dimensional **column vector**. This means that $\mb{y} = \Phi(X) \mb{w}$ is a $n$-dimensional vector$, as we should expect. 

Looking back at what we've done, what exactly changes? Well, clearly we can simply take $X \to \Phi(X)$, and so the likelihood is

$$ 
    p(\mb{y} | X, \mb{w}) \propto \exp\left(-\frac{1}{2 \sigma_n^2} \left|\mb{y} - \Phi(X) \mb{w}\right|^2  - \frac{1}{2} \mb{w}^\top \Sigma_p^{-1} \mb{w}\right)\,,
$$

or

$$
    p(\mb{y} | X, \mb{w}) \propto \exp\left(-\frac{1}{2 \sigma_n^2} \left|y_i - \Phi_i^a w_a\right|^2  - \frac{1}{2} w_a \Sigma_{p,ab}^{-1} w_b \right)
$$

with associated posterior density

$$
    p(\mb{w} | \mb{y}, X) = \mathcal{N}(\bar{\mb{w}}, A^{-1})\,.
$$

Again, we define the mean of this posterior as

$$
    \overline{\mb{w}} = \frac{1}{\sigma_n^2} \left(\Sigma_p^{-1} + \frac{1}{\sigma_n^2} \Phi^\top(X) \Phi(X)\right)^{-1} \Phi^\top(X) \mb{y}
$$

and its covariance matrix

$$
A = \frac{1}{\sigma_n^2} \Phi^\top(X) \Phi(X) + \Sigma_p^{-1}\,,
$$

both having more-or-less the same form as the linear regression example. 

Furthermore, if we are only interested in making predictions then the predictive distribution is simply

$$
    p(f_* | \mb{x}_*, \mb{y}, X) = \mathcal{N}(\sigma_n^{-2} \bm{\phi}_* A^{-1} \Phi(x)^\top \mb{y}, \bm{\phi}_*^\top A^{-1} \bm{\phi}_*)\,,
$$

where $\bm{\phi}_* \equiv \phi(\mb{x}_*)$ is a $d$-dimensional vector.

Now, let's consider what this means. $\Phi(X)$, as written, is a $n \times d$ matrix, where $d$ is the number of basis functions, parameters or the dimensionality of feature space, while $n$ is the number of points in $X$. We must then invert $A$, an $d \times d$ matrix, everytime we want to make predictions, which can be pretty bad if $d \gg n$, that is, the number of parameters or the dimension of feature space is much larger than the number of points.

But no one says we have to invert this matrix. For instance, if we are interested in predictions, what we really need to compute $\bm{\phi}_*^\top A^{-1} \bm{\phi}_*$, which is a single number. There's a bunch of ways to compute a single number, and if we are clever about it we can avoid inverting massive matrices. To be clever, we have to reorganize 

$$ 
    \bm{\phi}_*^\top A^{-1} \bm{\phi}_* = \bm{\phi}_*^\top \left(\sigma_n^{-2} \Phi^\top \Phi + \Sigma_p^{-1}\right)^{-1} \bm{\phi}_*\,.
$$

The basic idea is to group terms so that we only have to invert an $n \times n$ matrix. If we squint, we can see the way forward. $\Sigma_p$ is a $d \times d$ matrix and $\Phi$ is a $n \times d$ matrix, so $K \equiv \Phi \Sigma_p \Phi^\top$ is a $n \times n$ matrix.

We are saved by a matrix identity. Note that

$$
\left(Z + U W V^\top\right)^{-1} = Z^{-1} - Z^{-1} U \left(W^{-1} + V^\top Z^{-1} U\right)^{-1} V^\top Z^{-1}
$$

and so if we identify $Z = \Sigma_p^{-1}$, $U = V = \sigma_n^{-1} \Phi^\top$, and $W = \mathbb{I}$, then

$$\bm{\phi}_*^\top \left(\sigma_n^{-2} \Phi^\top \Phi + \Sigma_p^{-1}\right)^{-1} \bm{\phi}_* = \bm{\phi}_*^\top \Sigma_p \bm{\phi}_* - \bm{\phi}_*^\top \Sigma_p \Phi(K + \sigma_n^2 \mathbb{I})^{-1} \Phi^\top \Sigma_p \bm{\phi}_* $$

and we are only inverting the $n \times n$ matrix $K + \sigma_n^2 \mathbb{I}$. We see that

$$
    \sigma_n^{-2} \Phi^\top (K + \sigma_n^2 \mathbb{I}) = \sigma_n^2 \Phi^\top (\Phi \Sigma_p \Phi^\top + \sigma_n^2 \mathbb{I}) = A \Sigma_p \Phi^\top
$$


This allows us to rewrite the predictive density as

$$
p(f_* | \mb{x}_*, X, \mb{y}) = \mathcal{N} \!\left(\bm{\phi}_* \Sigma_p \Phi^\top \right)
$$


