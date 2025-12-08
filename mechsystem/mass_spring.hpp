#ifndef MASS_SPRING_HPP
#define MASS_SPRING_HPP

#include <nonlinfunc.hpp>
#include <timestepper.hpp>

using namespace ASC_ode;

#include <vector.hpp>
using namespace nanoblas;

#include <Eigen/Dense>

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

template <int D>
class MassSpringSystem
{
  std::vector<Fix<D>> m_fixes;
  std::vector<Mass<D>> m_masses;
  std::vector<Spring> m_springs;
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

  auto &fixes() { return m_fixes; }
  auto &masses() { return m_masses; }
  auto &springs() { return m_springs; }

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

// Constraint: g(x) = 0, with Jacobian G(x) = ∂g/∂x
class Constraint
{
public:
  size_t ndof; // total DOFs = D * #masses
  size_t ncon; // number of scalar constraints (m)

  Constraint(size_t ndof_, size_t ncon_)
      : ndof(ndof_), ncon(ncon_) {}

  // g(x)
  virtual void evaluateG(VectorView<double> x,
                         VectorView<double> gx) const = 0;

  // G(x) = ∂g/∂x
  virtual void evaluateJacobian(VectorView<double> x,
                                MatrixView<double> G) const = 0;

  virtual ~Constraint() = default;
};

// Example: keep distance between mass A and B equal to L0
template <int D>
class DistanceConstraint : public Constraint
{
  size_t massA;
  size_t massB;
  double L0;

public:
  DistanceConstraint(size_t ndof, size_t _massA, size_t _massB, double _L0)
      : Constraint(ndof, 1), massA(_massA), massB(_massB), L0(_L0) {}

  virtual void evaluateG(VectorView<double> x,
                         VectorView<double> gx) const override
  {
    gx = 0.0;

    auto xmat = x.asMatrix(ndof / D, D); // rows = masses, cols = coordinates
    Vec<D> pA = xmat.row(massA);
    Vec<D> pB = xmat.row(massB);

    Vec<D> diff = pA - pB;
    double dist = norm(diff);
    double dist2 = dist * dist; // ||pA - pB||^2

    // g(x) = ||pA - pB||^2 - L0^2
    gx(0) = dist2 - L0 * L0;
  }

  virtual void evaluateJacobian(VectorView<double> x,
                                MatrixView<double> G) const override
  {
    G = 0.0;

    auto xmat = x.asMatrix(ndof / D, D);
    Vec<D> pA = xmat.row(massA);
    Vec<D> pB = xmat.row(massB);
    Vec<D> diff = pA - pB;

    // ∂g/∂pA = 2 diff, ∂g/∂pB = -2 diff
    for (int d = 0; d < D; ++d)
    {
      size_t colA = massA * D + d;
      size_t colB = massB * D + d;
      G(0, colA) = 2.0 * diff(d);
      G(0, colB) = -2.0 * diff(d);
    }
  }
};

template <int D>
class MSS_Function : public NonlinearFunction
{
  MassSpringSystem<D> &mss;
  Constraint *constraint; // if this is null -> unconstrained, not null -> constrained

public:
  // Unconstrained constructor (exactly like old version)
  MSS_Function(MassSpringSystem<D> &_mss)
      : mss(_mss), constraint(nullptr) {}

  // Constrained constructor (with Lagrange multiplier)
  MSS_Function(MassSpringSystem<D> &_mss, Constraint &_constr)
      : mss(_mss), constraint(&_constr) {}

  virtual size_t dimX() const override { return D * mss.masses().size(); }
  virtual size_t dimF() const override { return D * mss.masses().size(); }

  virtual void evaluate(VectorView<double> x, VectorView<double> f) const override
  {
    const size_t ndof = dimX();

    // Compute forces F(x) as before
    Vector<> F(ndof);
    F = 0.0;
    auto xmat = x.asMatrix(mss.masses().size(), D);
    auto Fmat = F.asMatrix(mss.masses().size(), D);

    // Gravity forces:
    for (size_t i = 0; i < mss.masses().size(); i++)
      Fmat.row(i) = mss.masses()[i].mass * mss.getGravity();

    // Springs:
    for (auto spring : mss.springs())
    {
      auto [c1, c2] = spring.connectors;
      Vec<D> p1, p2;
      if (c1.type == Connector::FIX)
        p1 = mss.fixes()[c1.nr].pos;
      else
        p1 = xmat.row(c1.nr);
      if (c2.type == Connector::FIX)
        p2 = mss.fixes()[c2.nr].pos;
      else
        p2 = xmat.row(c2.nr);

      double force = spring.stiffness * (norm(p1 - p2) - spring.length);
      Vec<D> dir12 = 1.0 / norm(p1 - p2) * (p2 - p1);
      if (c1.type == Connector::MASS)
        Fmat.row(c1.nr) += force * dir12;
      if (c2.type == Connector::MASS)
        Fmat.row(c2.nr) -= force * dir12;
    }

    // If there is no constraint: old behavior (unconstrained)
    if (!constraint)
    {
      // f = acceleration = M^-1*F
      auto fmat = f.asMatrix(mss.masses().size(), D);
      for (size_t i = 0; i < mss.masses().size(); i++)
        fmat.row(i) = (1.0 / mss.masses()[i].mass) * Fmat.row(i);
      return;
    }

    // With constraint: solve system for [a; λ] with a linear solver, then only give a to Newmark solver
    const size_t m = constraint->ncon; // number of constraints

    // Build mass matrix M (diagonal)
    Matrix<> Mmat(ndof, ndof);
    Mmat = 0.0;
    for (size_t i = 0; i < mss.masses().size(); ++i)
      for (int d = 0; d < D; ++d)
      {
        size_t k = i * D + d;
        Mmat(k, k) = mss.masses()[i].mass;
      }

    // Compute g(x) and G(x) (with G(x) being the Jacobian of g(x))
    Vector<> gx(m);
    Matrix<> G(m, ndof);
    constraint->evaluateG(x, gx);       // gx = g(x)
    constraint->evaluateJacobian(x, G); // G = ∂g/∂x

    // Assemble matrix A with M on the diagonal block, G^T in the top-right block, G in the bottom-left block
    // Assemble right-hand side with force vector F(x) on top and −g(x) at the bottom
    Matrix<> A(ndof + m, ndof + m);
    Vector<> rhs(ndof + m);
    A = 0.0;

    // Top-left block: M
    for (size_t i = 0; i < ndof; ++i)
      for (size_t j = 0; j < ndof; ++j)
        A(i, j) = Mmat(i, j);

    // Top-right block: G^T; bottom-left block: G
    for (size_t r = 0; r < m; ++r)
      for (size_t c = 0; c < ndof; ++c)
      {
        A(c, ndof + r) = G(r, c); // G^T
        A(ndof + r, c) = G(r, c); // G
      }

    // RHS = [F; -g(x)]
    for (size_t i = 0; i < ndof; ++i)
      rhs(i) = F(i);
    for (size_t j = 0; j < m; ++j)
      rhs(ndof + j) = -gx(j);

    // Solve A * sol = rhs using Eigen
    Eigen::MatrixXd AE(ndof + m, ndof + m);
    Eigen::VectorXd bE(ndof + m);

    // copy A and rhs into Eigen structures
    for (size_t i = 0; i < ndof + m; ++i)
    {
      bE(i) = rhs(i);
      for (size_t j = 0; j < ndof + m; ++j)
        AE(i, j) = A(i, j);
    }

    // solve: [a; λ] = AE^{-1} * bE
    Eigen::VectorXd xE = AE.fullPivLu().solve(bE);

    // Extract acceleration a into f to be able to give it to the Newmark solver
    for (size_t i = 0; i < ndof; ++i)
      f(i) = xE(i);
  }


  virtual void evaluateDeriv(VectorView<double> x, MatrixView<double> df) const override
  {

    df = 0.0;

    auto xmat = x.asMatrix(mss.masses().size(), D);

    auto GetPos = [&](const Connector &c) -> Vec<D> {
      return (c.type ==Connector::FIX) ? mss.fixes()[c.nr].pos : xmat.row(c.nr);
    };

    auto AddBlock = [&](size_t row_idx, size_t col_index, const Matrix<D, D>& block){
      for (size_t i = 0; i < D; i++)
        for (size_t j = 0; j < count; j++)
          df(row_idx * D +i, col_index * D + j) += block(i,j);
    };


    for (auto spring : mss.springs())
    {
      auto [c1, c2] = spring.connectors;

      // Get Positions
      Vec<D> d = GetPos(c2) - GetPos(c1);
      double L = norm(d);
      double L0 = spring.length;

      Vec<D> n = d / L

      double k_elastic = spring.stiffness;
      double k_geometric = spring.stiffness * (L - L0) / L;

      Matrix<D, D> K;
      for (size_t i = 0; i < D; i++)
        for (size_t j = 0; j < D; j++)
          K(i, j) = k_geometric * (i==j) + (k_elastic - k_geometric) * n(i) * n(j);
      
      if (c1.type == Connector::MASS) {
        AddBlock(c1.nr, c1.nr,  -1.0 * K);
        if (c2.type == Connector::MASS)
          AddBlock(c1.nr, c2.nr, K)
      }
      if (c2.type == Connector::MASS) {
        AddBlock(c2.nr, c2.nr, -1.0 * K);
        if (c1.type == Connector::MASS)
          AddBlock(c2.nr, c1.nr, K)
      }
    }
  

    for (size_t i = 0; i < mss.constraints().size(); i++)
    {
      const auto & d_const = mss.constraints()[k];
      Vec<D> d = GetPos(c2) - GetPos(c1);
      double L = norm(d);
      Vec<D> n = d / L
      size_t l_idx = D * mss.masses().size() + k;
      double lambda = x(l_idx)

      Matrix<D, D> H;
      double scale = lambda / L;
      for (size_t i = 0; i++)
        for (size_t j = 0; j++)
          H(i, j) = scale * ((i==j) - n(i) * n(j))
      
      if (c1.type == Connector::MASS) {
        AddBlock(c1.nr, c1.nr,  -1.0 * H);
        if (c2.type == Connector::MASS)
          AddBlock(c1.nr, c2.nr, H)
      }
      if (c2.type == Connector::MASS) {
        AddBlock(c2.nr, c2.nr, -1.0 * H);
        if (c1.type == Connector::MASS)
          AddBlock(c2.nr, c1.nr, H)
      }


    }
  }

};

#endif
