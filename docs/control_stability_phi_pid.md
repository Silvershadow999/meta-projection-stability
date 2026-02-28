# Control Stability Evidence — φ-Quasi-PID (Discrete-Time, Conservative)

## 1) Local linearized model (time-varying / aperiodic)
We consider a local linearization of the risk/error dynamics around an operating point:

\[
x_{k+1} = A_k x_k + B_k u_k
\]

- \(x_k\): local error state (e.g. \(x_k = r_k - r^\*\), with \(r^\*=0.5\))
- \(u_k\): controller output (PID + aperiodic correction)
- \(A_k\): aperiodic transition matrix (quasi-crystalline / time-varying)

Define worst-case spectral radius bound:

\[
\lambda_{\max} := \sup_k \rho(A_k)
\]

where \(\rho(\cdot)\) is the spectral radius. In practice we may use a conservative bound via a compatible norm.

## 2) φ-Quasi-PID controller (dominant gain)
The controller is (discrete) PID with an aperiodic finite correction:

\[
u_k = K_p e_k + K_i \sum_{j=0}^k e_j \Delta t + K_d \frac{e_k-e_{k-1}}{\Delta t} + c_k
\]

In our spec-inspired parameterization:

- \(K_p \approx \varphi\)
- \(K_i \approx 1/\varphi\)
- \(K_d \approx \varphi^2\)

For a conservative stability condition we focus on the dominant gain contribution (worst-case):

\[
g \sim \varphi^2
\]

## 3) Closed-loop stability (discrete-time)
A standard sufficient condition for asymptotic stability in discrete time is:

\[
\sup_k \rho(A_k^{cl}) < 1
\]

Conservatively, model the controller effect as scaling by a worst-case gain \(g\):

\[
\rho(A_k^{cl}) \;\lesssim\; g \cdot \rho(A_k)
\]

Thus a sufficient robust condition is:

\[
g \cdot \lambda_{\max} < 1
\]

Substituting \(g=\varphi^2\):

\[
\varphi^2 \cdot \lambda_{\max} < 1
\quad\Longleftrightarrow\quad
\varphi^2 < \frac{1}{\lambda_{\max}}
\]

This matches the repo/spec stability condition used as a conservative brake.

## 4) Aperiodic Penrose-style correction converges
The correction is implemented as a finite approximation of a geometrically-scaled sum:

\[
c_k = \sum_{n=0}^{N-1} \left(\frac{1}{\varphi}\right)^n \Delta p_{k,n}
\]

Since \(|1/\varphi| \approx 0.618 < 1\), the infinite series is absolutely convergent (bounded inputs), and the finite approximation is numerically stable.

## 5) Fail-safe damping on violation
If the conservative stability condition is violated, the implementation applies a safety brake:

- damp controller output by a factor (e.g. `* 0.5`)

This is standard gain-scheduling / emergency damping to push the effective closed-loop gain back toward the stable region.

## 6) Implementation note (current limitation)
The bound \(\lambda_{\max}\) must be estimated in production (e.g. via local Jacobian / transition estimation).
Until then, the repo uses a conservative placeholder and treats this as an engineering safeguard, not a formal global proof.

