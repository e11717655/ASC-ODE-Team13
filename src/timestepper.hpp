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
    virtual void DoStep(double tau, VectorView<double> y) = 0;
  };

  class ExplicitEuler : public TimeStepper
  {
    Vector<> m_vecf;
  public:
    ExplicitEuler(std::shared_ptr<NonlinearFunction> rhs) 
    : TimeStepper(rhs), m_vecf(rhs->dimF()) {}
    void DoStep(double tau, VectorView<double> y) override
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

    void DoStep(double tau, VectorView<double> y) override
    {
      m_yold->set(y);
      m_tau->set(tau);
      NewtonSolver(m_equ, y);
    }
  };
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

} 

#endif