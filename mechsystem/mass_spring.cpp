#include "mass_spring.hpp"
#include "Newmark.hpp"

int main()
{
  MassSpringSystem<2> mss;
  mss.setGravity( {0,-9.81} );
  
  // 1. Setup Objects (Correct Way using Fix)
  // ---------------------------------------------------------
  // Pivot: Use a proper Fix object
  // This returns a Connector of type FIX
  auto fPivot = mss.addFix( { { 0.0, 0.0 } } ); 

  // Mass A: Connector of type MASS
  auto mA = mss.addMass( { 1.0, { 1.0, 0.0 } } );

  // Mass B: Connector of type MASS
  auto mB = mss.addMass( { 1.0, { 2.0, 0.0 } } );
  // ---------------------------------------------------------

  std::cout << "mss: " << std::endl << mss << std::endl;

  double tend  = 10;
  double steps = 1000;
  
  // NDOF = 2 masses * 2 coordinates = 4 DOFs
  // (The Fix object is not part of the state vector x)
  size_t ndof = 2 * mss.masses().size();

  Vector<> x(ndof);
  Vector<> dx(ndof);
  Vector<> ddx(ndof);

  mss.getState (x, dx, ddx);

  // 2. Add Constraints
  // ---------------------------------------------------------
  // Constraint 1: Pivot (Fix) <-> Mass A | Length = 1.0
  // Note: We pass the Connectors (fPivot, mA) and the 'mss' reference
  // so the constraint can look up the Fix position internally.
  auto c1 = std::make_shared<DistanceConstraint<2>>(ndof, fPivot, mA, 1.0, mss);
  mss.addConstraint(c1);

  // Constraint 2: Mass A <-> Mass B | Length = 1.0
  auto c2 = std::make_shared<DistanceConstraint<2>>(ndof, mA, mB, 1.0, mss);
  mss.addConstraint(c2);
  // ---------------------------------------------------------

  // 3. Solve
  auto mss_func = std::make_shared<MSS_Function<2>>(mss);
  auto mass = std::make_shared<IdentityFunction> (ndof);

  SolveODE_Newmark(tend, steps, x, dx,  mss_func, mass,
                   [](double t, VectorView<double> x) {
                     // Print positions
                     // Mass A is now at index 0,1 (since Fix is not in x)
                     // Mass B is now at index 2,3
                     std::cout << "t=" << t 
                               << " | A=(" << x(0) << "," << x(1) << ")"
                               << " | B=(" << x(2) << "," << x(3) << ")" 
                               << std::endl;
                   });
}