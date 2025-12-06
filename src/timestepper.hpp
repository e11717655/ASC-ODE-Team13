#ifndef TIMERSTEPPER_HPP
#define TIMERSTEPPER_HPP

#include <functional>
#include <exception>

#include "Newton.hpp"
#include <memory>


namespace ASC_ode
{
  
  class TimeStepper
  { 
  protected:
    std::shared_ptr<NonlinearFunction> m_rhs;
  public:
    TimeStepper(std::shared_ptr<NonlinearFunction> rhs) : m_rhs(rhs) {}
    virtual ~TimeStepper() = default;
    virtual void doStep(double tau, VectorView<double> y) = 0;
  };

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

    void doStep(double tau, VectorView<double> y) override
    {
      m_yold->set(y);
      m_tau->set(tau);
      NewtonSolver(m_equ, y);
    }
  };

  class ImprovedEuler : public TimeStepper
  {
    Vector<> m_vecf;
    Vector<> m_ytilde;
  public:
    ImprovedEuler(std::shared_ptr<NonlinearFunction> rhs) 
    : TimeStepper(rhs), m_vecf(rhs->dimF()), m_ytilde(rhs->dimX()) {}

    void DoStep(double tau, VectorView<double> y) override
    {
      this->m_rhs->evaluate(y, m_vecf);
      m_ytilde = y;
      m_ytilde +=  (0.5*tau) * m_vecf;

      this->m_rhs->evaluate(m_ytilde, m_vecf);
      y += tau * m_vecf;
    }
  };
class CrankNicolson : public TimeStepper
{
  std::shared_ptr<NonlinearFunction> m_equ;
  std::shared_ptr<Parameter> m_tau;
  std::shared_ptr<ConstantFunction> m_explicit_part;
  
  // Pre-allocate temporary vectors to save speed
  Vector<> m_vecf; 
  Vector<> m_helper_vec; 

public:
  CrankNicolson(std::shared_ptr<NonlinearFunction> rhs) 
    : TimeStepper(rhs), 
      m_tau(std::make_shared<Parameter>(0.0)),
      m_vecf(rhs->dimF()),       
      m_helper_vec(rhs->dimX()) 
  {
    m_explicit_part = std::make_shared<ConstantFunction>(rhs->dimX()); // <--- FIXED: Added semicolon
    auto ynew = std::make_shared<IdentityFunction>(rhs->dimX());

    // FIXED: Removed (0.5 * m_tau).
    // We define the equation generically as: y_new - explicit_part - parameter * f(y_new)
    // We will set 'parameter' to be (0.5 * tau) later in DoStep.
    m_equ = ynew - m_explicit_part - m_tau * m_rhs; 
  }

  void DoStep(double tau, VectorView<double> y) override
  {
    // 1. Evaluate f(y_n)
    m_rhs->evaluate(y, m_vecf);

    // 2. Calculate explicit part: y + (0.5 * tau) * f(y_n)
    m_helper_vec = y; 
    m_helper_vec += (0.5 * tau) * m_vecf;

    // 3. Update equation constants
    m_explicit_part->set(m_helper_vec);

    // FIXED: We set the parameter value to be half the step size here.
    m_tau->set(0.5 * tau); 

    // 4. Solve
    NewtonSolver(m_equ, y);
  }
};

}


#endif
