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
  enum CONTYPE { FIX = 1, MASS = 2 };
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

// --- Constraint Base Class ---
class Constraint
{
public:
  size_t ndof;
  size_t ncon;

  Constraint(size_t ndof_, size_t ncon_) : ndof(ndof_), ncon(ncon_) {}

  virtual void evaluateG(VectorView<double> x, VectorView<double> gx) const = 0;
  virtual void evaluateJacobian(VectorView<double> x, MatrixView<double> G) const = 0;
  
  // NEW: Add scaled Hessian matrix to H_accum: H_accum += scale * H(x)
  virtual void addHessian(VectorView<double> x, double scale, MatrixView<double> H_accum) const = 0;
  
  // NEW: Compute Hessian * vector product: result += H(x) * v
  virtual void applyHessian(VectorView<double> x, VectorView<double> v, VectorView<double> result) const = 0;

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

  Connector addFix(Fix<D> p) { m_fixes.push_back(p); return {Connector::FIX, m_fixes.size() - 1}; }
  Connector addMass(Mass<D> m) { m_masses.push_back(m); return {Connector::MASS, m_masses.size() - 1}; }
  size_t addSpring(Spring s) { m_springs.push_back(s); return m_springs.size() - 1; }
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
    for (size_t i = 0; i < m_masses.size(); i++) {
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
    for (size_t i = 0; i < m_masses.size(); i++) {
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
  for (auto f : mss.fixes()) ost << f.pos << std::endl;
  ost << "masses: " << std::endl;
  for (auto m : mss.masses()) ost << "m = " << m.mass << ", pos = " << m.pos << std::endl;
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
  Vec<D> GetPos(const Connector &c, const MatrixView<double>& xmat) const {
      if (c.type == Connector::FIX) return mss.fixes()[c.nr].pos;
      else return xmat.row(c.nr);
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

    for (int d = 0; d < D; ++d) {
      if (c1.type == Connector::MASS) G(0, c1.nr * D + d) =  2.0 * diff(d);
      if (c2.type == Connector::MASS) G(0, c2.nr * D + d) = -2.0 * diff(d);
    }
  }

  virtual void addHessian(VectorView<double> x, double scale, MatrixView<double> H) const override
  {
      // Hessian of dist^2 is block matrix with 2*I
      for(int d=0; d<D; d++) {
          if(c1.type == Connector::MASS) H(c1.nr*D+d, c1.nr*D+d) += 2.0 * scale;
          if(c2.type == Connector::MASS) H(c2.nr*D+d, c2.nr*D+d) += 2.0 * scale;
          
          if(c1.type == Connector::MASS && c2.type == Connector::MASS) {
              H(c1.nr*D+d, c2.nr*D+d) -= 2.0 * scale;
              H(c2.nr*D+d, c1.nr*D+d) -= 2.0 * scale;
          }
      }
  }

  virtual void applyHessian(VectorView<double> x, VectorView<double> v, VectorView<double> res) const override
  {
      auto vmat = v.asMatrix(this->ndof / D, D);
      auto resmat = res.asMatrix(this->ndof / D, D);
      
      Vec<D> v1 = (c1.type==Connector::MASS) ? vmat.row(c1.nr) : Vec<D>(0.0);
      Vec<D> v2 = (c2.type==Connector::MASS) ? vmat.row(c2.nr) : Vec<D>(0.0);
      Vec<D> diff = 2.0 * (v1 - v2);
      
      if(c1.type == Connector::MASS) resmat.row(c1.nr) += diff;
      if(c2.type == Connector::MASS) resmat.row(c2.nr) -= diff;
  }
};

// --- Function Evaluator ---
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
    // ... (This function remains largely the same as the previous correct KKT version)
    const size_t ndof = dimX();

    Vector<> F(ndof); F = 0.0;
    auto xmat = x.asMatrix(mss.masses().size(), D);
    auto Fmat = F.asMatrix(mss.masses().size(), D);

    // Forces
    for (size_t i = 0; i < mss.masses().size(); i++)
      Fmat.row(i) = mss.masses()[i].mass * mss.getGravity();

    for (auto spring : mss.springs()) {
      auto [c1, c2] = spring.connectors;
      Vec<D> p1 = (c1.type == Connector::FIX) ? mss.fixes()[c1.nr].pos : xmat.row(c1.nr);
      Vec<D> p2 = (c2.type == Connector::FIX) ? mss.fixes()[c2.nr].pos : xmat.row(c2.nr);
      double dist = norm(p1 - p2);
      double force = spring.stiffness * (dist - spring.length);
      Vec<D> dir = (dist > 1e-12) ? (1.0 / dist) * (p2 - p1) : Vec<D>(0.0);
      if (c1.type == Connector::MASS) Fmat.row(c1.nr) += force * dir;
      if (c2.type == Connector::MASS) Fmat.row(c2.nr) -= force * dir;
    }

    size_t m = 0;
    for(auto &c : mss.constraints()) m += c->ncon;

    if (m == 0) {
      for (size_t i = 0; i < ndof; i++) f(i) = F(i) / mss.masses()[i/D].mass;
      return;
    }

    Matrix<> Mmat(ndof, ndof); Mmat = 0.0;
    for (size_t i = 0; i < mss.masses().size(); ++i)
      for (int d = 0; d < D; ++d) Mmat(i * D + d, i * D + d) = mss.masses()[i].mass;

    Vector<> gx(m); Matrix<> G(m, ndof); gx = 0.0; G = 0.0;

    size_t row = 0;
    for(auto &c : mss.constraints()) {
        VectorView<double> sub_gx(c->ncon, &gx(row));
        MatrixView<double> sub_G(c->ncon, ndof, &G(row, 0));
        c->evaluateG(x, sub_gx);
        c->evaluateJacobian(x, sub_G);
        row += c->ncon;
    }

    Matrix<> A(ndof + m, ndof + m); Vector<> rhs(ndof + m); A = 0.0;

    for (size_t i = 0; i < ndof; ++i) A(i, i) = Mmat(i, i);
    for (size_t r = 0; r < m; ++r)
      for (size_t c = 0; c < ndof; ++c) { A(c, ndof + r) = G(r, c); A(ndof + r, c) = G(r, c); }

    double beta = 10000.0; 
    for (size_t i = 0; i < ndof; ++i) rhs(i) = F(i);
    for (size_t j = 0; j < m; ++j)    rhs(ndof + j) = -beta * gx(j);

    Eigen::MatrixXd AE(ndof + m, ndof + m);
    Eigen::VectorXd bE(ndof + m);
    for (size_t i = 0; i < ndof + m; ++i) {
      bE(i) = rhs(i);
      for (size_t j = 0; j < ndof + m; ++j) AE(i, j) = A(i, j);
    }
    
    Eigen::VectorXd xE = AE.fullPivLu().solve(bE);
    for (size_t i = 0; i < ndof; ++i) f(i) = xE(i);
  }

  // --- EXACT ANALYTIC DERIVATIVE ---
  virtual void evaluateDeriv(VectorView<double> x, MatrixView<double> df) const override
  {
    // 1. We must RE-SOLVE the system to get the Lagrange Multipliers (lambda) and Acceleration (a)
    //    because `evaluate` didn't output them.
    //solve 
    const size_t ndof = dimX();
    size_t m = 0;
    for(auto &c : mss.constraints()) m += c->ncon;
    
    // ... Reconstruct Matrices (Copied logic from evaluate) ...
    // Note: In production code, factorize this into a helper method!
    
    Vector<> F(ndof); F = 0.0;
    auto xmat = x.asMatrix(mss.masses().size(), D);
    auto Fmat = F.asMatrix(mss.masses().size(), D);
    for (size_t i = 0; i < mss.masses().size(); i++) Fmat.row(i) = mss.masses()[i].mass * mss.getGravity();

    // Calculate Spring Stiffness Matrix (K_spring) while we are at it
    Matrix<> K_spring(ndof, ndof); K_spring = 0.0;
    
    for (auto spring : mss.springs()) {
      auto [c1, c2] = spring.connectors;
      Vec<D> p1 = (c1.type == Connector::FIX) ? mss.fixes()[c1.nr].pos : xmat.row(c1.nr);
      Vec<D> p2 = (c2.type == Connector::FIX) ? mss.fixes()[c2.nr].pos : xmat.row(c2.nr);
      double dist = norm(p1 - p2);
      double force = spring.stiffness * (dist - spring.length);
      Vec<D> dir = (dist > 1e-12) ? (1.0 / dist) * (p2 - p1) : Vec<D>(0.0);
      
      // Force accumulation
      if (c1.type == Connector::MASS) Fmat.row(c1.nr) += force * dir;
      if (c2.type == Connector::MASS) Fmat.row(c2.nr) -= force * dir;

      // Stiffness Matrix Calculation
      if(dist > 1e-12) {
          Vec<D> n = dir;
          double k_e = spring.stiffness;
          double k_g = spring.stiffness * (1.0 - spring.length / dist);
          Matrix<double> K_loc(D, D);
          for(int i=0; i<D; i++) for(int j=0; j<D; j++) 
              K_loc(i,j) = (i==j ? k_g : 0.0) + (k_e - k_g)*n(i)*n(j);
              
          if(c1.type==Connector::MASS && c2.type==Connector::MASS) {
              for(int i=0; i<D; i++) for(int j=0; j<D; j++) {
                  K_spring(c1.nr*D+i, c1.nr*D+j) -= K_loc(i,j); // Note: K usually on LHS, here on RHS
                  K_spring(c2.nr*D+i, c2.nr*D+j) -= K_loc(i,j);
                  K_spring(c1.nr*D+i, c2.nr*D+j) += K_loc(i,j);
                  K_spring(c2.nr*D+i, c1.nr*D+j) += K_loc(i,j);
              }
          } else if (c1.type==Connector::MASS) {
               for(int i=0; i<D; i++) for(int j=0; j<D; j++) K_spring(c1.nr*D+i, c1.nr*D+j) -= K_loc(i,j);
          } else if (c2.type==Connector::MASS) {
               for(int i=0; i<D; i++) for(int j=0; j<D; j++) K_spring(c2.nr*D+i, c2.nr*D+j) -= K_loc(i,j);
          }
      }
    }

    if (m == 0) {
        // No constraints: J = M^-1 * K
        for(size_t i=0; i<ndof; i++) for(size_t j=0; j<ndof; j++) 
            df(i,j) = K_spring(i,j) / mss.masses()[i/D].mass;
        return;
    }

    // --- Constrained Derivative Logic ---
    // 1. Solve for State (a, lambda)
    Matrix<> Mmat(ndof, ndof); Mmat = 0.0;
    for(size_t i=0; i<ndof; i++) Mmat(i,i) = mss.masses()[i/D].mass;
    
    Vector<> gx(m); Matrix<> G(m, ndof); gx = 0.0; G = 0.0;
    size_t row = 0;
    for(auto &c : mss.constraints()) {
        VectorView<double> sub_gx(c->ncon, &gx(row));
        MatrixView<double> sub_G(c->ncon, ndof, &G(row, 0));
        c->evaluateG(x, sub_gx);
        c->evaluateJacobian(x, sub_G);
        row += c->ncon;
    }

    Matrix<> A_KKT(ndof + m, ndof + m); A_KKT = 0.0;
    for(size_t i=0; i<ndof; i++) A_KKT(i,i) = Mmat(i,i);
    for(size_t r=0; r<m; r++) for(size_t c=0; c<ndof; c++) { 
        A_KKT(c, ndof+r) = G(r, c); A_KKT(ndof+r, c) = G(r, c); 
    }
    
    Eigen::MatrixXd AE(ndof+m, ndof+m);
    Eigen::VectorXd bE(ndof+m);
    double beta = 10000.0;
    
    for(int i=0; i<ndof+m; i++) for(int j=0; j<ndof+m; j++) AE(i,j) = A_KKT(i,j);
    for(int i=0; i<ndof; i++) bE(i) = F(i);
    for(int i=0; i<m; i++)    bE(ndof+i) = -beta * gx(i);
    
    Eigen::VectorXd sol = AE.fullPivLu().solve(bE);
    Vector<> acc(ndof);
    Vector<> lambda(m);
    for(int i=0; i<ndof; i++) acc(i) = sol(i);
    for(int i=0; i<m; i++)    lambda(i) = sol(ndof+i);

    // 2. Build Derivative System RHS
    // We want to solve KKT * [da/dx; dlambda/dx] = RHS_matrix
    // RHS Top Block = K_spring - sum(lambda * H_g)
    // RHS Bot Block = - (beta * G + H_g * a)
    
    // A. Geometric Stiffness of Constraints (H_lambda)
    Matrix<> H_lambda(ndof, ndof); H_lambda = 0.0;
    row = 0;
    for(auto &c : mss.constraints()) {
        // H_accum += (-lambda) * H(c)  (Note: lambda is on LHS in M*a + G*l = F, so deriv term G*dl + dG*l = ... -> dG*l moves to RHS as -dG*l)
        // Wait: The term is G^T * lambda. d(G^T * lambda) = (dG^T)*lambda + ...
        // So we need to subtract Hessian weighted by lambda.
        for(int k=0; k<c->ncon; k++) {
            c->addHessian(x, -lambda(row+k), H_lambda); 
        }
        row += c->ncon;
    }

    // B. Constraint Hessian times Acceleration (H_acc)
    // Term is d(G*a) = dG * a + G * da. 
    // dG * a vector needs to be computed.
    Vector<> H_acc_vec(ndof); // Helper to store result
    
    Eigen::MatrixXd RHS_mat(ndof + m, ndof);
    RHS_mat.setZero();

    // Fill Top Block: K_spring + H_lambda
    for(int i=0; i<ndof; i++) for(int j=0; j<ndof; j++) 
        RHS_mat(i,j) = K_spring(i,j) + H_lambda(i,j);

    // Fill Bottom Block: - (beta * G + dG * a)
    // beta * G part:
    for(int r=0; r<m; r++) for(int c=0; c<ndof; c++) 
        RHS_mat(ndof+r, c) = -beta * G(r,c);
        
    // dG * a part:
    row = 0;
    for(auto &c : mss.constraints()) {
        // For each constraint equation k, we calculate (dG_k * a)^T -> which is a row in the bottom block.
        // Actually dG_k * a is a vector (result of Hessian * a). 
        // This vector becomes the row in the Jacobian of the constraints?
        // Let's rely on symmetry. The term is [ Hessian(g) * a ]^T
        for(int k=0; k<c->ncon; k++) {
             H_acc_vec = 0.0;
             // Only passing 'acc' (acceleration) to applyHessian
             c->applyHessian(x, acc, H_acc_vec); // H_acc_vec = H(g_k) * a
             
             for(int j=0; j<ndof; j++) {
                 RHS_mat(ndof+row+k, j) -= H_acc_vec(j);
             }
        }
        row += c->ncon;
    }

    // 3. Solve for Jacobian
    // [ da/dx ] = AE_inv * RHS_mat
    Eigen::MatrixXd J_total = AE.fullPivLu().solve(RHS_mat);

    // 4. Extract da/dx (Top-Left Block)
    for(int i=0; i<ndof; i++) for(int j=0; j<ndof; j++)
        df(i,j) = J_total(i,j);
  }
};

#endif