# NSSC1 Numerical Methods for Ordinary Differential Equations Exercises Part 1

For this exercise, we compared several different time-stepping methods for a mass attached to a spring, this is described by the ODE:

$$m y''(t) = -k y(t)$$

giving us a simple harmonic oscilator system, with $k$ as the spring constant, $m$ as the mass, and $y(t)$ as displacement.
This second-order equation can be replaced with the first order system:

$$
y_{0}^{\prime} = y_{1} \\
y_{1}^{\prime} = - \frac{k}{m} y_{0}
$$

```cpp
class MassSpring : public NonlinearFunction
{
  double m_mass;
  double m_stiffness;
  
public:
  MassSpring (double m, double k)
    : m_mass(m), m_stiffness(k) {}
  
  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }
  
  void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    f(0) = x(1);
    f(1) = -m_stiffness/m_mass * x(0);
  }
  
  void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    df = 0.0;
    df(0,1) = 1;
    df(1,0) = -m_stiffness/m_mass;
  }
};

```

Once this system has been set up they can be solved using various timestep methods.

These methods are implemented by modifying the base class `TimeStepper`.

```cpp

class TimeStepper
{
  protected:
    std::shared_ptr<NonlinearFunction> m_rhs;
  public:
    TimeStepper(std::shared_ptr<NonlinearFunction> rhs) : m_rhs(rhs) {}
    virtual ~TimeStepper() = default;
    virtual void doStep(double tau, VectorView<double> y) = 0;
};
```

# Explicit Euler Method
The first time step method is the Explicit Euler Method. It is an explicit, one-step method where the ODE

$$
y^{\prime}(t) = f(y(t)) \quad \forall t \in [0,T], \quad y(0) = y_{0}
$$

is solved by using this method:

$$
y_{i+1} = y_{i} + \tau f(y_{i}) \quad \text{for } 0 \leq i < n
$$

with $\tau = T/n$. The error of this method is proptional to the step size, making it have first order accuracy $O(\tau)$. The method is only conditionally stable, and when utilized for a pure harmonic oscilator, as it was in this case, it is not stable as energy is added to the system causing the amplitude of the oscillations to increase.

The class `ExplicitEuler` looks like this:

```cpp

class ExplicitEuler : public TimeStepper
{
  Vector<> m_vecf;
public:
  ExplicitEuler(std::shared_ptr<NonlinearFunction> rhs)
  : TimeStepper(rhs), m_vecf(rhs->dimF()) {}
  
  void doStep(double tau, VectorView<double> y) override
  {
    this->m_rhs->evaluate(y, m_vecf);
    y += tau * m_vecf;
  }
};
```

Here the base class is updated by modifying the `doStep` function. It now takes `tau` and `m_vectf` as inputs and then updates the state vector by adding them.

# Implicit Euler Method
Next up, we had to try out the Implicit Euler Method. Compared to the explicit Euler method, the implicit Euler Method is more computationally expensive, due to it having to solve a potentially nonlinear equation for $y_i+1$. This is done using a built in Newton Solver. The method is also first-order accurate, with the error decreasing linearly with step size $\tau$. However, this method is both A-stable and L-stable. It has a dampening effect built in which causes the oscillations to decay over time, despite not having any friction defined in the system.


The Method is defined by

$$
\frac{y_{i+1} - y_i}{t_{i+1} - t_i} = f(t_{i+1}, y_{i+1})
$$

or it can be rewritten as

$$
y_{i+1} = y_i + \tau\, f(t_{i+1}, y_{i+1}), \qquad 0 \le i < n
$$

The code was already provided. The implementation utilized was

$$
y - y_{old} - τf(y) = 0
$$

with the provided class

```cpp
class ImplicitEuler : public TimeStepper
  {
    std::shared_ptr<NonlinearFunction> m_equ;
    std::shared_ptr<Parameter> m_tau;
    std::shared_ptr<ConstantFunction> m_yold;
  public:
    ImplicitEuler(std::shared_ptr<NonlinearFunction> rhs)
    : TimeStepper(rhs), m_tau(std::make_shared<Parameter>(0.0))
    {
      m_yold = std::make_shared<ConstantFunction>(rhs->dimX());
      auto ynew = std::make_shared<IdentityFunction>(rhs->dimX());
      m_equ = ynew - m_yold - m_tau * m_rhs;
    }

    void DoStep(double tau, VectorView<double> y) override
    {
      m_yold->set(y);
      m_tau->set(tau);
      NewtonSolver(m_equ, y);
    }
  };
```
# Improved Euler Method

This method involved modifying the `ExplicitEuler` time-stepping method by replacing the time steps with

$$
\tilde{y} = y _n + \frac{τ}{2}f(y_n) \\
y_{n+1} = y_{n} + τf(\tilde{y})
$$

This method is also explicit, however it is second-order $O(τ^{2}))$ accurate, causing the error to decrease in proportion to the square of the timestep. It is still not stable, and also does not need a `NewtonSolver` as it only requires two function evaluations.

```cpp
class ImprovedEuler : public TimeStepper{
    Vector<> m_vecf;
    Vector<> halfway;
  public:
    ImprovedEuler(std::shared_ptr<NonlinearFunction> rhs)
    : TimeStepper(rhs), m_vecf(rhs->dimF()), halfway(rhs->dimX()) {}
    void DoStep(double tau, VectorView<double> y) override
    {
      this->m_rhs->evaluate(y, m_vecf);
      halfway = y;
      halfway += (0.5*tau)*m_vecf;

      m_rhs->evaluate(halfway, m_vecf);
      y += tau * m_vecf;
    }
  };
```

# Crank-Nicolson Method

The `CrankNicolson` class was implemented by modifying the `ImplicitEuler` class. They both rely on the `NewtonStepper` as they are implicit methods, the difference however, is in the way the steps are calculated

$$
y_{i+1} = y_i + \frac{τ}{2}(f(t_i, y_i) + f(t_{i+1}, y_{i+1}) \qquad 0 \le i < n
$$

This method is a one-step method. It utilizes the trapezoidal integration rule for the formulation of the steps and has second-order accuracy. It is A-stable, and unlike the implicit Euler Method it is energy conserving, meaning that the oscillations will continue without decay or explostion (Explicit Euler).

```cpp
class CrankNicolson : public TimeStepper
  {
    std::shared_ptr<NonlinearFunction> m_equ;
    std::shared_ptr<Parameter> m_tau;
    std::shared_ptr<ConstantFunction> m_yold;
    std::shared_ptr<ConstantFunction> m_f_old;

  public:
    CrankNicolson(std::shared_ptr<NonlinearFunction> rhs)
    : TimeStepper(rhs), m_tau(std::make_shared<Parameter>(0.0))
    {
      m_yold = std::make_shared<ConstantFunction>(rhs->dimX());
      auto ynew = std::make_shared<IdentityFunction>(rhs->dimX());
      m_fold = std::make_shared<ConstantFunction>(rhs->dimX());
      m_equ = ynew - m_yold - 0.5 * (m_tau * (m_rhs + m_fold));
    }

    void DoStep(double tau, VectorView<double> y) override
    {
      Vector<double> fy_val(y.size());
      m_rhs->evaluate(y, fy_val);
      
      m_fold->set(fy_val);
      m_yold->set(y);       
      m_tau->set(tau);      

      NewtonSolver(m_equ, y);
    }
  };
```


### Interpretation of Plots

10 Steps

<img src="../results/phase_plot_10_12.57.png" width="350px"> <img src="../results/time_evo_10_12.57.png" width="350px">

Phase portrait and time evolution for 10 steps, end time 12.57.

<img src="../results/phase_plot_10_62.83.png" width="350px"> <img src="../results/time_evo_10_62.83.png" width="350px">

Phase portrait and time evolution for 10 steps, end time 62.83.

<img src="../results/phase_plot_10_314.16.png" width="350px"> <img src="../results/time_evo_10_314.16.png" width="350px">

Phase portrait and time evolution for 10 steps, end time 314.16.

1. **Large step sizes (τ = 10)**  
   For large time steps, the numerical method introduces significant deformation in the phase plot.  
   The the phase plots become heavily distorted, showing numerical damping or energy increase.
   It is a direct consequence of the low resolution of the discretization and instability of the method.

100 Steps

<img src="../results/phase_plot_100_12.57.png" width="350px"> <img src="../results/time_evo_100_12.57.png" width="350px">

Phase portrait and time evolution for 100 steps, end time 12.57.

<img src="../results/phase_plot_100_62.83.png" width="350px"> <img src="../results/time_evo_100_62.83.png" width="350px">

Phase portrait and time evolution for 100 steps, end time 62.83.

<img src="../results/phase_plot_100_314.16.png" width="350px"> <img src="../results/time_evo_100_314.16.png" width="350px">

Phase portrait and time evolution for 100 steps, end time 314.16.

1000 Steps

<img src="../results/phase_plot_1000_12.57.png" width="350px"> <img src="../results/time_evo_1000_12.57.png" width="350px">

Phase portrait and time evolution for 1000 steps, end time 12.57.

<img src="../results/phase_plot_1000_62.83.png" width="350px"> <img src="../results/time_evo_1000_62.83.png" width="350px">

Phase portrait and time evolution for 1000 steps, end time 62.83.

<img src="../results/phase_plot_1000_314.16.png" width="350px"> <img src="../results/time_evo_1000_314.16.png" width="350px">

Phase portrait and time evolution for 1000 steps, end time 314.16.

2. **Medium step sizes (τ = 100, 1000)**  
   As the time step decreases, the phase portraits become smoother and more symmetric. In the plots we can nicely see, that the explicit Euler Method increases in amplitude, while the implicit Euler Method decreases over time. This is especially noticable in the plots where the End Time is set to 314.16 seconds.

10000 steps

<img src="../results/phase_plot_10000_12.57.png" width="350px"> <img src="../results/time_evo_10000_12.57.png" width="350px">

Phase portrait and time evolution for 10000 steps, end time 12.57.

<img src="../results/phase_plot_10000_62.83.png" width="350px"> <img src="../results/time_evo_10000_62.83.png" width="350px">

Phase portrait and time evolution for 10000 steps, end time 62.83.

<img src="../results/phase_plot_10000_314.16.png" width="350px"> <img src="../results/time_evo_10000_314.16.png" width="350px">

Phase portrait and time evolution for 10000 steps, end time 314.16.

100000 steps

<img src="../results/phase_plot_100000_12.57.png" width="350px"> <img src="../results/time_evo_100000_12.57.png" width="350px">

Phase portrait and time evolution for 100000 steps, end time 12.57.

<img src="../results/phase_plot_100000_62.83.png" width="350px"> <img src="../results/time_evo_100000_62.83.png" width="350px">

Phase portrait and time evolution for 100000 steps, end time 62.83.

<img src="../results/phase_plot_100000_314.16.png" width="350px"> <img src="../results/time_evo_100000_314.16.png" width="350px">

Phase portrait and time evolution for 100000 steps, end time 314.16.

3. **Very small step sizes (τ = 10000, 100000)**  
   For the smallest step size considered, the phase curves converge toward the analytical solution.  
   The trajectories become closed and circular, indicating stable long term behavior and good energy conservation. Altough on both plots with End Time = 314.16s, the explicit Euler Method again increases in Amplitude, showing that even at extremely small step sizes the method is not energy-preserving and accumulates a systematic energy error over long integration times.

   ### Overall Conclusions

As we can see in the plots, for very large τ, the waveform becomes visibly jagged. Even if we increase the number of steps, some time evolution plots show growing amplitude, others show damped amplitude, depending on the Numeric Method used. For example the explicit Euler method injects energy, therefore resulting in a growing amplitude. In contrast,the implicit Euler Method is both A-stable and L-stable. It has built in dampening, which dissipates energy and causes dampening over time.
The Crank-Nicholson Method is energy preserving, therefore the oscillations will continue without dampening or amplification. 


For τ = 100000, the numerical time evolution is smooth and closely matches the expected sinusoidal behavior.  This demonstrates the expected convergence behavior of the integrators: as τ decreases, the numerical error shrinks at the rate predicted by the theoretical order of the method.


# NSSC1 Numerical Methods for Ordinary Differential Equations Exercises Part 2
<img src="../docs/image.png" width="350px">

## Exercise Report: Electric Network Simulation

### 1. Mathematical Model

We model a simple RC circuit consisting of a resistor $R$, a capacitor $C$, and a time-dependent voltage source $U_0(t) = \cos(100 \pi t)$.

Using Ohm's law ($U_R = R I$) and the capacitor equation ($I = C \dot{U}_C$), we apply Kirchhoff's voltage law, which states that voltages around a loop sum to zero ($U_0 = U_R + U_C$). This yields the linear Ordinary Differential Equation (ODE):

$$U_C(t) + RC \frac{dU_C}{dt}(t) = U_0(t)$$

Rearranging for the derivative, we obtain the standard non-autonomous form $\dot{y} = f(t, y)$:

$$\frac{dU_C}{dt} = \frac{U_0(t) - U_C(t)}{RC} = \frac{\cos(100 \pi t) - U_C}{RC}$$

### 2. Transformation to Autonomous Form

To utilize numerical solvers designed for autonomous systems ($\dot{\mathbf{y}} = F(\mathbf{y})$), we must eliminate the explicit time dependence. We achieve this by extending the state space to treat time as a dependent variable.

We define a new state vector $\mathbf{y} \in \mathbb{R}^2$:

* $y_1 = U_C$ (Capacitor Voltage)
* $y_2 = t$ (Time)

The derivatives are formulated as follows:

* **Voltage:** $\dot{y}_1 = \frac{U_0(y_2) - y_1}{RC}$
* **Time:** $\dot{y}_2 = \frac{dt}{dt} = 1$

This results in the following autonomous system:

$$\dot{\mathbf{y}} = \begin{pmatrix} \frac{\cos(100 \pi y_2) - y_1}{RC} \\ 1 \end{pmatrix}$$

### 3. Jacobian for Implicit Methods

For the implementation of implicit solvers (Implicit Euler, Crank-Nicolson), the Jacobian matrix $J = \frac{\partial F}{\partial \mathbf{y}}$ is required. Differentiating the system vector with respect to $y_1$ and $y_2$:

$$J = \begin{pmatrix} 
\frac{\partial \dot{y}_1}{\partial y_1} & \frac{\partial \dot{y}_1}{\partial y_2} \\
\frac{\partial \dot{y}_2}{\partial y_1} & \frac{\partial \dot{y}_2}{\partial y_2}
\end{pmatrix} = 
\begin{pmatrix} 
-\frac{1}{RC} & -\frac{100 \pi \sin(100 \pi y_2)}{RC} \\
0 & 0
\end{pmatrix}$$

The simulation uses this system with initial conditions $\mathbf{y}(0) = [0, 0]^T$.

### 4. Setup

When running the code 3 different stepping sizes have been used. $N = \{100, 1000, 10000\}$. And the two different model parameters $R=1, C=1$ and $R=100, C=10^{-6}$. As a total time 0.1 seconds were choosen. The results displayed in graphs can be found below. For each step size the resulting Capacity Voltage has been plotted for the 3 numerical methods: The explicit Euler, the implicit Euler and the Crank-Nicolson. The input voltage has been scaled to the the size of the capacity voltage, so that the phase shift can be nicely seen.

**Simulation Results**

<img src="../results/Electrical_Circuit/circuit_plot_100_1.png" width="350px"> 

*Plot for 100 time steps and $R=1, C=1$*

<img src="../results/Electrical_Circuit/circuit_plot_1000_1.png" width="350px">

*Plot for 1000 time steps and $R=1, C=1$*

<img src="../results/Electrical_Circuit/circuit_plot_10000_1.png" width="350px">

*Plot for 10000 time steps and $R=1, C=1$*

<img src="../results/Electrical_Circuit/circuit_plot_100_1.png" width="350px"> 

*Plot for 100 time steps and $R=100, C=10^{-6}$*

<img src="../results/Electrical_Circuit/circuit_plot_1000_1.png" width="350px">

*Plot for 1000 time steps and $R=100, C=10^{-6}$*

<img src="../results/Electrical_Circuit/circuit_plot_10000_1.png" width="350px">

*Plot for 10000 time steps and $R=100, C=10^{-6}$*

---

## Analysis of Simulation Results

### Part 1: ($R=100, C=10^{-6}$)

**1. Stiffness & Stability Limits**

The system is stiff, characterized by a fast time constant $\tau = RC = 10^{-4}s$ and a large negative eigenvalue $\lambda = -10000$.

* **Explicit Euler Stability Condition:** $|1 + \lambda \Delta t| \le 1$.
    For this system, stability requires $\Delta t \le 2 \cdot 10^{-4}s$.
* **Implicit Method Stability:** Implicit Euler and Crank-Nicolson are A-stable, meaning they remain stable for any step size $\Delta t$, regardless of $\tau$.

**2. Method Behavior**

* **Explicit Euler**
    * **100 Steps ($\Delta t = 10\tau$):** Violates the stability condition ($10\tau > 2\tau$). The solution explodes immediately, reaching physically impossible values ($10^{57}$).
    * **1000 Steps ($\Delta t = \tau$):** Stable. Because $\Delta t = \tau$, the derivative projection lands exactly on the steady-state value in a single step (jumping from 0 to 1 instantly).

* **Implicit Euler**
    * **Stable but Damped:** Even with large steps ($10\tau$), the solution never explodes. However, it exhibits numerical damping, causing the voltage to lag behind the rapid initial changes.

* **Crank-Nicolson**
    * **Stable but Oscillatory:** Also stable for large steps ($10\tau$). However, it suffers from spurious oscillations (overshoot $>1$) during the initial transient because it averages the derivatives at $t_n$ and $t_{n+1}$, struggling to resolve the fast charge-up.

### Part 2: System ($R=1, C=1$)

**1. System Characteristics**

With the new parameters, the system dynamics change drastically:

* Time Constant: $\tau = RC = 1 \, s$.
* Eigenvalue: $\lambda = -1$.
* Stability Limit (Explicit): $|1 - \Delta t| \le 1 \Rightarrow \Delta t \le 2 \, s$.

This system is non-stiff because the time constant ($\tau=1$) is much slower than the simulation time steps used ($10^{-3}$ to $10^{-5}$).

**2. Stability Analysis**

Unlike the previous case, all methods are stable for all step sizes tested.
The largest step size used is $\Delta t = 0.001 \, s$.
Since $0.001 \ll 2$ (the stability limit), Explicit Euler is well within its stability region.
*Observation:* No explosions or wild oscillations are observed.

**3. Physical Behavior: Low-Pass Filtering & Phase Shift**

The circuit acts as a strong low-pass filter with significant phase lag.

* **Amplitude Attenuation:**
    Since the source frequency $\omega = 100\pi \approx 314$ rad/s is much higher than the cutoff $\omega_c = 1$ rad/s, the amplitude is attenuated by a factor of $\approx 1/314$. The data shows peaks around $0.003$V, matching this theory.

* **Phase Shift ($\phi$):**
    The theoretical phase shift is $\phi = -\arctan(\omega RC) = -\arctan(314) \approx -89.8^\circ$.

* **Observation in Data:**
    The voltage should lag the source by almost exactly $90^\circ$ ($\pi/2$), which corresponds to a time lag of $0.005s$ (quarter period). Looking at the graphs the capacitor voltage behaves like a Sine wave while the source is a Cosine wave, demonstrating the predicted $90^\circ$ lag.

### Comparison of Methods

Because the step sizes are small relative to the system time constant ($\Delta t \ll \tau$), the numerical errors are small for all methods.

* **Amplitude Discrepancy (Low N):** At the coarsest resolution (100 steps), there is a distinct difference in amplitude due to numerical error properties.
    * Explicit Euler overestimates the peak ($\approx 0.0036$V).
    * Crank-Nicolson sits in the middle ($\approx 0.0031$V).
    * Implicit Euler is the most damped and underestimates the peak ($\approx 0.0026$V).
    * *Reason:* This occurs because Explicit Euler extrapolates linearly (overshooting convex curves), while Implicit Euler is numerically dissipative.

* **Resolution:** Even the coarsest resolution (100 steps, $\Delta t = 0.001$) provides $\sim 20$ points per period of the source ($T=0.02s$), sufficient to capture the general wave shape, though the amplitude accuracy varies by method as noted above.

### Summary Comparison

| Parameter Set | Time Constant $\tau$ | System Type | Explicit Euler (100 steps) | Physical Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Set 1 ($R=100, C=10^{-6}$)** | $10^{-4} s$ | Stiff | Unstable (Explodes) | Fast charge (tracks source) |
| **Set 2 ($R=1, C=1$)** | $1 s$ | Non-Stiff | Stable | Filtered (Low amp, $90^\circ$ lag) |

## Code

```cpp
class ElectricNetwork : public NonlinearFunction
{
private:
  double R;
  double C; 

public:
  ElectricNetwork(double r, double c) : R(r), C(c) {}

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }

  void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    double Uc = x(0);
    double t  = x(1);
    f(0) = (std::cos(100.0 * M_PI * t) - Uc) / (R * C);
    f(1) = 1.0;
  }

  void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    double t = x(1);
    df = 0.0;
    df(0,0) = -1.0 / (R * C);
    df(0,1) = -(100.0 * M_PI) * std::sin(100.0 * M_PI * t) / (R * C);
    df(1,0) = 0.0;
    df(1,1) = 0.0;
  }
};