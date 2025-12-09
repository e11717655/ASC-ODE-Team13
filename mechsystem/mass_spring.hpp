#ifndef MASS_SPRING_HPP
#define MASS_SPRING_HPP

#include <nonlinfunc.hpp>
#include <timestepper.hpp>
#include <vector.hpp>
#include <Eigen/Dense>
#include <vector>
#include <memory>

using namespace ASC_ode;
using namespace nanoblas;
template <int D>
class Mass
{
public:
  double mass;
  Vec<D> pos;
  Vec<D> vel = 0.0;
  Vec<D> acc = 0.0;
};

template <int D>
class Fix
{
public:
  Vec<D> pos;
};

class Connector
{
public:
  enum CONTYPE
  {
    FIX = 1,
    MASS = 2
  };
  CONTYPE type;
  size_t nr;
};

std::ostream &operator<<(std::ostream &ost, const Connector &con)
{
  ost << "type = " << int(con.type) << ", nr = " << con.nr;
  return ost;
}

class Spring
{
public:
  double length;
  double stiffness;
  std::array<Connector, 2> connectors;
};

//Constraint Base Class
class Constraint
{
public:
  size_t ndof;
  size_t ncon;

  Constraint(size_t ndof_, size_t ncon_) : ndof(ndof_), ncon(ncon_) {}

  virtual void evaluateG(VectorView<double> x, VectorView<double> gx) const = 0;
  virtual void evaluateJacobian(VectorView<double> x, MatrixView<double> G) const = 0;
  virtual ~Constraint() = default;
};

template <int D>
class MassSpringSystem
{
  std::vector<Fix<D>> m_fixes;
  std::vector<Mass<D>> m_masses;
  std::vector<Spring> m_springs;
  std::vector<std::shared_ptr<Constraint>> m_constraints;
  Vec<D> m_gravity = 0.0;

public:
  void setGravity(Vec<D> gravity) { m_gravity = gravity; }
  Vec<D> getGravity() const { return m_gravity; }

  Connector addFix(Fix<D> p)
  {
    m_fixes.push_back(p);
    return {Connector::FIX, m_fixes.size() - 1};
  }
  Connector addMass(Mass<D> m)
  {
    m_masses.push_back(m);
    return {Connector::MASS, m_masses.size() - 1};
  }
  size_t addSpring(Spring s)
  {
    m_springs.push_back(s);
    return m_springs.size() - 1;
  }
  void addConstraint(std::shared_ptr<Constraint> c) { m_constraints.push_back(c); }

  auto &fixes() { return m_fixes; }
  auto &masses() { return m_masses; }
  auto &springs() { return m_springs; }
  auto &constraints() { return m_constraints; }

  void getState(VectorView<> values, VectorView<> dvalues, VectorView<> ddvalues)
  {
    auto valmat = values.asMatrix(m_masses.size(), D);
    auto dvalmat = dvalues.asMatrix(m_masses.size(), D);
    auto ddvalmat = ddvalues.asMatrix(m_masses.size(), D);
    for (size_t i = 0; i < m_masses.size(); i++)
    {
      valmat.row(i) = m_masses[i].pos;
      dvalmat.row(i) = m_masses[i].vel;
      ddvalmat.row(i) = m_masses[i].acc;
    }
  }

  void setState(VectorView<> values, VectorView<> dvalues, VectorView<> ddvalues)
  {
    auto valmat = values.asMatrix(m_masses.size(), D);
    auto dvalmat = dvalues.asMatrix(m_masses.size(), D);
    auto ddvalmat = ddvalues.asMatrix(m_masses.size(), D);
    for (size_t i = 0; i < m_masses.size(); i++)
    {
      m_masses[i].pos = valmat.row(i);
      m_masses[i].vel = dvalmat.row(i);
      m_masses[i].acc = ddvalmat.row(i);
    }
  }
};

template <int D>
std::ostream &operator<<(std::ostream &ost, MassSpringSystem<D> &mss)
{
  ost << "fixes:" << std::endl;
  for (auto f : mss.fixes())
    ost << f.pos << std::endl;
  ost << "masses: " << std::endl;
  for (auto m : mss.masses())
    ost << "m = " << m.mass << ", pos = " << m.pos << std::endl;
  ost << "springs: " << std::endl;
  for (auto sp : mss.springs())
    ost << "length = " << sp.length << ", stiffness = " << sp.stiffness
        << ", C1 = " << sp.connectors[0] << ", C2 = " << sp.connectors[1] << std::endl;
  return ost;
}

// --- Distance Constraint Implementation ---
template <int D>
class DistanceConstraint : public Constraint
{
  Connector c1;
  Connector c2;
  double L0;
  MassSpringSystem<D> &mss;

public:
  DistanceConstraint(size_t ndof, Connector _c1, Connector _c2, double _L0, MassSpringSystem<D> &_mss)
      : Constraint(ndof, 1), c1(_c1), c2(_c2), L0(_L0), mss(_mss) {}

  // Helper to get Position
  Vec<D> GetPos(const Connector &c, const MatrixView<double> &xmat) const
  {
    if (c.type == Connector::FIX)
      return mss.fixes()[c.nr].pos;
    else
      return xmat.row(c.nr);
  }

  virtual void evaluateG(VectorView<double> x, VectorView<double> gx) const override
  {
    auto xmat = x.asMatrix(this->ndof / D, D);
    Vec<D> diff = GetPos(c1, xmat) - GetPos(c2, xmat);
    gx(0) = dot(diff, diff) - L0 * L0;
  }

  virtual void evaluateJacobian(VectorView<double> x, MatrixView<double> G) const override
  {
    G = 0.0;
    auto xmat = x.asMatrix(this->ndof / D, D);
    Vec<D> diff = GetPos(c1, xmat) - GetPos(c2, xmat);

    for (int d = 0; d < D; ++d)
    {
      if (c1.type == Connector::MASS)
        G(0, c1.nr * D + d) = 2.0 * diff(d);
      if (c2.type == Connector::MASS)
        G(0, c2.nr * D + d) = -2.0 * diff(d);
    }
  }
};
//Function Evaluatortemplate <int D>
template <int D>
class MSS_Function : public NonlinearFunction
{
  MassSpringSystem<D> &mss;

public:
  MSS_Function(MassSpringSystem<D> &_mss) : mss(_mss) {}

  virtual size_t dimX() const override { return D * mss.masses().size(); }
  virtual size_t dimF() const override { return D * mss.masses().size(); }

  virtual void evaluate(VectorView<double> x, VectorView<double> f) const override
  {
    //This function remains largely the same as the previous correct KKT version
    const size_t ndof = dimX();

    Vector<> F(ndof);
    F = 0.0;
    auto xmat = x.asMatrix(mss.masses().size(), D);
    auto Fmat = F.asMatrix(mss.masses().size(), D);

    // Forces
    for (size_t i = 0; i < mss.masses().size(); i++)
      Fmat.row(i) = mss.masses()[i].mass * mss.getGravity();

    for (auto spring : mss.springs())
    {
      auto [c1, c2] = spring.connectors;
      Vec<D> p1 = (c1.type == Connector::FIX) ? mss.fixes()[c1.nr].pos : xmat.row(c1.nr);
      Vec<D> p2 = (c2.type == Connector::FIX) ? mss.fixes()[c2.nr].pos : xmat.row(c2.nr);
      double dist = norm(p1 - p2);
      double force = spring.stiffness * (dist - spring.length);
      Vec<D> dir = (dist > 1e-12) ? (1.0 / dist) * (p2 - p1) : Vec<D>(0.0);
      if (c1.type == Connector::MASS)
        Fmat.row(c1.nr) += force * dir;
      if (c2.type == Connector::MASS)
        Fmat.row(c2.nr) -= force * dir;
    }

    size_t m = 0;
    for (auto &c : mss.constraints())
      m += c->ncon;

    if (m == 0)
    {
      for (size_t i = 0; i < ndof; i++)
        f(i) = F(i) / mss.masses()[i / D].mass;
      return;
    }

    Matrix<> Mmat(ndof, ndof);
    Mmat = 0.0;
    for (size_t i = 0; i < mss.masses().size(); ++i)
      for (int d = 0; d < D; ++d)
        Mmat(i * D + d, i * D + d) = mss.masses()[i].mass;

    Vector<> gx(m);
    Matrix<> G(m, ndof);
    gx = 0.0;
    G = 0.0;

    size_t row = 0;
    for (auto &c : mss.constraints())
    {
      VectorView<double> sub_gx(c->ncon, &gx(row));
      MatrixView<double> sub_G(c->ncon, ndof, &G(row, 0));
      c->evaluateG(x, sub_gx);
      c->evaluateJacobian(x, sub_G);
      row += c->ncon;
    }

    Matrix<> A(ndof + m, ndof + m);
    Vector<> rhs(ndof + m);
    A = 0.0;

    for (size_t i = 0; i < ndof; ++i)
      A(i, i) = Mmat(i, i);
    for (size_t r = 0; r < m; ++r)
      for (size_t c = 0; c < ndof; ++c)
      {
        A(c, ndof + r) = G(r, c);
        A(ndof + r, c) = G(r, c);
      }

    double beta = 10000.0;
    for (size_t i = 0; i < ndof; ++i)
      rhs(i) = F(i);
    for (size_t j = 0; j < m; ++j)
      rhs(ndof + j) = -beta * gx(j);

    Eigen::MatrixXd AE(ndof + m, ndof + m);
    Eigen::VectorXd bE(ndof + m);
    for (size_t i = 0; i < ndof + m; ++i)
    {
      bE(i) = rhs(i);
      for (size_t j = 0; j < ndof + m; ++j)
        AE(i, j) = A(i, j);
    }

    Eigen::VectorXd xE = AE.fullPivLu().solve(bE);
    for (size_t i = 0; i < ndof; ++i)
      f(i) = xE(i);
  }


  virtual void evaluateDeriv(VectorView<double> x, MatrixView<double> df) const override
  {
    const size_t ndof = dimX();
    size_t m = 0;
    for (auto &c : mss.constraints())
      m += c->ncon;

    //Force and Stiffness (Standard Mass-Spring)
    auto xmat = x.asMatrix(mss.masses().size(), D);
    

    Matrix<> K_spring(ndof, ndof);
    K_spring = 0.0;

    for (auto spring : mss.springs())
    {
      auto [c1, c2] = spring.connectors;
      Vec<D> p1 = (c1.type == Connector::FIX) ? mss.fixes()[c1.nr].pos : xmat.row(c1.nr);
      Vec<D> p2 = (c2.type == Connector::FIX) ? mss.fixes()[c2.nr].pos : xmat.row(c2.nr);
      double dist = norm(p1 - p2);
      
      // We don't strictly need the Force vector F anymore, just the Stiffness K
      if (dist > 1e-12)
      {
        Vec<D> dir = (1.0 / dist) * (p2 - p1);
        double k_e = spring.stiffness;
        double k_g = spring.stiffness * (1.0 - spring.length / dist);
        
        Matrix<double> K_loc(D, D);
        for (int i = 0; i < D; i++)
          for (int j = 0; j < D; j++)
            K_loc(i, j) = (i == j ? k_g : 0.0) + (k_e - k_g) * dir(i) * dir(j);

        // Assemble K_spring (Negative sign because K = -dF/dx)
        if (c1.type == Connector::MASS) 
           for (int i=0; i<D; i++) for(int j=0; j<D; j++) K_spring(c1.nr*D+i, c1.nr*D+j) -= K_loc(i,j);
        
        if (c2.type == Connector::MASS) 
           for (int i=0; i<D; i++) for(int j=0; j<D; j++) K_spring(c2.nr*D+i, c2.nr*D+j) -= K_loc(i,j);
        
        if (c1.type == Connector::MASS && c2.type == Connector::MASS) {
           for (int i=0; i<D; i++) for(int j=0; j<D; j++) {
              K_spring(c1.nr*D+i, c2.nr*D+j) += K_loc(i,j);
              K_spring(c2.nr*D+i, c1.nr*D+j) += K_loc(i,j);
           }
        }
      }
    }

    //Case: Unconstrained
    if (m == 0)
    {
      for (size_t i = 0; i < ndof; i++)
        for (size_t j = 0; j < ndof; j++)
          df(i, j) = K_spring(i, j) / mss.masses()[i / D].mass;
      return;
    }

    Eigen::MatrixXd AE(ndof + m, ndof + m);
    AE.setZero();

    //Fill Mass Matrix
    for (size_t i = 0; i < ndof; i++)
      AE(i, i) = mss.masses()[i / D].mass;

    //Fill Constraint Jacobian G
    Matrix<> G(m, ndof);
    G = 0.0;
    size_t row = 0;
    for (auto &c : mss.constraints())
    {
      MatrixView<double> sub_G(c->ncon, ndof, &G(row, 0));
      c->evaluateJacobian(x, sub_G);
      
      for (size_t r = 0; r < c->ncon; r++) {
        for (size_t col = 0; col < ndof; col++) {
           AE(ndof + row + r, col) = G(row + r, col); // G
           AE(col, ndof + row + r) = G(row + r, col); // G^T
        }
      }
      row += c->ncon;
    }

    //Build RHS for Derivative
    Eigen::MatrixXd RHS_deriv(ndof + m, ndof);
    RHS_deriv.setZero();

    //Top rows: Just K_spring
    for (int i = 0; i < ndof; i++)
        for (int j = 0; j < ndof; j++)
            RHS_deriv(i, j) = K_spring(i, j);

    //Solve: AE * J = RHS
    Eigen::MatrixXd J_total = AE.fullPivLu().solve(RHS_deriv);

    //Copy result back to df
    for (int i = 0; i < ndof; i++)
      for (int j = 0; j < ndof; j++)
        df(i, j) = J_total(i, j);
  }
};
#endif 