#ifndef MASS_SPRING_HPP
#define MASS_SPRING_HPP

#include <nonlinfunc.hpp>
#include <timestepper.hpp>

using namespace ASC_ode;

#include <vector.hpp>
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

template <int D>
class MSS_Function : public NonlinearFunction
{
  MassSpringSystem<D> &mss;

public:
  MSS_Function(MassSpringSystem<D> &_mss)
      : mss(_mss) {}

  virtual size_t dimX() const override { return D * mss.masses().size(); }
  virtual size_t dimF() const override { return D * mss.masses().size(); }

  virtual void evaluate(VectorView<double> x, VectorView<double> f) const override
  {
    f = 0.0;

    auto xmat = x.asMatrix(mss.masses().size(), D);
    auto fmat = f.asMatrix(mss.masses().size(), D);

    for (size_t i = 0; i < mss.masses().size(); i++)
      fmat.row(i) = mss.masses()[i].mass * mss.getGravity();

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
        fmat.row(c1.nr) += force * dir12;
      if (c2.type == Connector::MASS)
        fmat.row(c2.nr) -= force * dir12;
    }

    for (size_t i = 0; i < mss.masses().size(); i++)
      fmat.row(i) *= 1.0 / mss.masses()[i].mass;
  }

  /**
   * Computes the exact Jacobian of the system acceleration.
   * J = d(Acceleration) / d(Position)
   */
virtual void evaluateDeriv(VectorView<double> x, MatrixView<double> df) const override
{
    // 1. Initialize Jacobian to zero
    df = 0.0;
    
    auto xmat = x.asMatrix(mss.masses().size(), D);

    // 2. Process Springs
    for (auto spring : mss.springs())
    {
        auto [c1, c2] = spring.connectors;
        
        // Get Positions
        Vec<D> p1, p2;
        if (c1.type == Connector::FIX) p1 = mss.fixes()[c1.nr].pos;
        else                           p1 = xmat.row(c1.nr);

        if (c2.type == Connector::FIX) p2 = mss.fixes()[c2.nr].pos;
        else                           p2 = xmat.row(c2.nr);

        // Constants
        double k = spring.stiffness;
        double L0 = spring.length;

        // Vectors
        Vec<D> diff = p2 - p1;
        double L = norm(diff);
        
        // Safety check
        if (L < 1e-14) continue;

        // Physics Coefficients
        // s_elastic: Resistance to stretching (k)
        // s_geometric: Resistance to rotation (Tension / L)
        double s_elastic = k;
        double s_geometric = k * (L - L0) / L;

        // Loop over dimensions (i=Row dimension, j=Col dimension)
        for (size_t i = 0; i < D; i++)
        {
            for (size_t j = 0; j < D; j++)
            {
                // Normalized direction components
                double n_i = diff(i) / L;
                double n_j = diff(j) / L;
                
                double delta = (i == j) ? 1.0 : 0.0;
                
                // Exact Derivative Formula
                // val = d(Force_i) / d(Pos_j)
                double val = s_geometric * delta + (s_elastic - s_geometric) * n_i * n_j;

                // --- MATRIX ASSEMBLY (with Mass Division) ---

                // 1. Mass 1 Diagonal (Effect of M1 on M1)
                if (c1.type == Connector::MASS)
                {
                    double m1 = mss.masses()[c1.nr].mass;
                    df(D * c1.nr + i, D * c1.nr + j) -= val / m1;
                }

                // 2. Mass 2 Diagonal (Effect of M2 on M2)
                if (c2.type == Connector::MASS)
                {
                    double m2 = mss.masses()[c2.nr].mass;
                    df(D * c2.nr + i, D * c2.nr + j) -= val / m2;
                }

                // 3. Off-Diagonals (Interaction)
                if (c1.type == Connector::MASS && c2.type == Connector::MASS)
                {
                    double m1 = mss.masses()[c1.nr].mass;
                    double m2 = mss.masses()[c2.nr].mass;

                    // Force on M1 due to M2
                    df(D * c1.nr + i, D * c2.nr + j) += val / m1;

                    // Force on M2 due to M1
                    df(D * c2.nr + i, D * c1.nr + j) += val / m2;
                }
            }
        }
    }
}
};

#endif
