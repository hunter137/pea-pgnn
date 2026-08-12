# Method

## Computational representation of incomplete knowledge

The package represents incomplete time-dependent engineering knowledge as

\[
\mathcal{K}=(\boldsymbol{\theta}^{p},\mathcal{F},\mathcal{C}),
\]

where \(\boldsymbol{\theta}^{p}\) contains quantitative parameter priors,
\(\mathcal{F}=\{f_k\}_{k=1}^{K}\) is a set of candidate temporal laws, and
\(\mathcal{C}\) contains qualitative structural properties.

The neural network receives a time-invariant condition vector \(\mathbf{x}\).
It learns bounded corrections to an empirical magnitude anchor \(A^p\) and an
empirical timescale anchor \(\tau^p\):

\[
A(\mathbf{x})=\operatorname{clip}\left(
A^p(1+\delta_A)+\delta_{\mathrm{add}}, A_{\min}, A_{\max}
\right),
\]

\[
\tau(\mathbf{x})=\operatorname{clip}\left(
\tau^p(1+\delta_\tau),\tau_{\min},\tau_{\max}
\right).
\]

The correction heads are initialized so that \(\delta_A=0\),
\(\delta_{\mathrm{add}}=0\), and \(\delta_\tau=0\). This makes the initial
parameterization genuinely anchor-centered even when the admissible relative
bounds are asymmetric.

## Candidate laws and convex mixture

For non-negative time \(t\), positive timescale \(\tau\), and positive exponent
\(\alpha\), the bundled candidate laws are

\[
f_1=\tanh\sqrt{t/\tau},
\qquad
f_2=\left(\frac{t}{t+\tau}\right)^\alpha,
\]

\[
f_3=\sqrt{\frac{t}{t+\tau^2/100}},
\qquad
f_4=\operatorname{clip}\left(
\frac{\log(1+t/\tau)}{\log(1+10^4)},0,1
\right).
\]

The network produces non-negative normalized weights
\(w_k(\mathbf{x})\) and forms

\[
g(t\mid\mathbf{x})=\sum_{k=1}^{K}w_k(\mathbf{x})f_k(t;\tau,\alpha),
\quad w_k\geq0,\quad\sum_kw_k=1.
\]

The point prediction is

\[
\widehat{y}(t\mid\mathbf{x})=A(\mathbf{x})g(t\mid\mathbf{x}).
\]

## Inherited properties

Every candidate law lies in \([0,1]\) and is non-decreasing in time over the
implemented domain. A convex mixture of these laws retains those properties.
Because the response magnitude is positive, the point predictor therefore
satisfies

\[
0\leq\widehat{y}(t\mid\mathbf{x})\leq A(\mathbf{x})
\]

and

\[
t_1\leq t_2\Rightarrow
\widehat{y}(t_1\mid\mathbf{x})\leq\widehat{y}(t_2\mid\mathbf{x}).
\]

This logic requires query time to remain outside the condition vector. If a
user inserts time into context, the network may change its weights or magnitude
between queries and the trajectory-level argument no longer applies.

## What is and is not guaranteed

The construction guarantees algebraic properties of the point predictor, not
predictive accuracy. It does not by itself justify extrapolation to unseen
materials, geometries, climates, databases, or missing-condition regimes. It
also does not guarantee that separately constructed uncertainty-interval
endpoints are monotone or bounded.

