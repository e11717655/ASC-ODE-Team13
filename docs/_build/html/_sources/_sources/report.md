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

Here the base class is updated by modifying the `doStep` function. It now takes `tau` and `m_vectf` as inputs and then updating the state vector by adding them.

# Implicit Euler Method

Next we tested the the Implicit euler Method. This method

Next up, we had to try out the Implicit Euler Method. Compared to the explicit Euler method, the implicit Euler Method is more computationally expensive, due to it having to solve a potentially nonlinear equation for $y_i+1$. This is done using a built in Newton Solver. The method is also first-order accurate, with the error decreasing linearly with step size $\tau$. However, this method is both A-stable and L-stable. It has a dampening effect built in which causes the oscillations to decay over time, despite not having any friction defined in the system.


The Method is defined by
\begin{eqnarray*}
\frac{y_{i+1} - y_i}{t_{i+1} - t_i} = f(t_{i+1}, y_{i+1})
\end{eqnarray*}

or it can be rewritten as

\begin{eqnarray*}
y_{i+1} = y_i + \tau\, f(t_{i+1}, y_{i+1}), \qquad 0 \le i < n
\end{eqnarray*}

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

This method involved modifying the the `ExplicitEuler` time-stepping method by replacing the time steps with

$$
\tilde{y} = y _n + \frac{τ}{2}f(y_n) \\
y_{n+1} = y_{n} + τf(\tilde{y})
$$

This method is also explicit, however it has a second-order $O(τ^{2}))$ accurate, causing the error to decrease in proportion to the square of the timestep. It is still not stable, and also does not need a `NewtonSolver` as it only requires two function evaluations.

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

This method is a one-step method. It utilizes the the trapezoidal integration rule for the formulation of the steps and has second-order accuracy. It is A-stable, and unlike the implicit Euler Method is energy conserving, meaning that the oscillations will continue without decay or explostion (Explicit Euler).

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
