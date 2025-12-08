#include "mass_spring.hpp"
#include "Newmark.hpp"

int main()
{
  MassSpringSystem<2> mss;
  mss.setGravity({0, -9.81});

  // 1. Setup Objects
  auto fPivot = mss.addFix({{0.0, 0.0}});
  auto mA = mss.addMass({1.0, {1.0, 0.0}});
  auto mB = mss.addMass({1.0, {2.0, 0.0}});

  std::cout << "mss: " << std::endl
            << mss << std::endl;

  double tend = 10;
  double steps = 1000;

  size_t ndof = 2 * mss.masses().size();

  Vector<> x(ndof);
  Vector<> dx(ndof);
  Vector<> ddx(ndof);

  mss.getState(x, dx, ddx);

  // 2. Add Constraints
  auto c1 = std::make_shared<DistanceConstraint<2>>(ndof, fPivot, mA, 1.0, mss);
  mss.addConstraint(c1);
  auto c2 = std::make_shared<DistanceConstraint<2>>(ndof, mA, mB, 1.0, mss);
  mss.addConstraint(c2);

  // 3. Solve
  auto mss_func = std::make_shared<MSS_Function<2>>(mss);
  auto mass = std::make_shared<IdentityFunction>(ndof);

  SolveODE_Newmark(tend, steps, x, dx, mss_func, mass,
                   [](double t, VectorView<double> x)
                   {
                     std::cout << "t=" << t
                               << " | A=(" << x(0) << "," << x(1) << ")"
                               << " | B=(" << x(2) << "," << x(3) << ")"
                               << std::endl;
                   });
}