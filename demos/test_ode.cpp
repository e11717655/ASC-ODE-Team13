#include <memory>
#include <iostream>
#include <fstream>
#include <cmath>

#include <nonlinfunc.hpp>
#include <timestepper.hpp>
#include <implicitRK.hpp>
#include <explicitRK.hpp>


using namespace ASC_ode;


class MassSpring : public NonlinearFunction
{
private:
  double mass;
  double stiffness;

public:
  MassSpring(double m, double k) : mass(m), stiffness(k) {}

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }
  
  void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    f(0) = x(1);
    f(1) = -stiffness/mass*x(0);
  }
  
  void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    df = 0.0;
    df(0,1) = 1;
    df(1,0) = -stiffness/mass;
  }
};





int main()
{
double tend = 4*M_PI;
  int steps = 100;
  double tau = tend/steps;

  Vector<> y = { 1, 0 };  // initializer list
  auto rhs = std::make_shared<MassSpring>(1.0, 1.0);



/*
  Vector<> Radau(3), RadauWeight(3);
  GaussRadau (Radau, RadauWeight);
  // not sure about weights, comput them via ComputeABfromC
  cout << "Radau = " << Radau << ", weight = " << RadauWeight <<  endl;
        Vector<> Gauss2c(2), Gauss3c(3);
*/
 

  // ExplicitEuler stepper(rhs);
  // ImplicitEuler stepper(rhs);

  // RungeKutta stepper(rhs, Gauss2a, Gauss2b, Gauss2c);

  // Gauss3c .. points tabulated, compute a,b:
  auto [Gauss3a,Gauss3b] = computeABfromC (Gauss3c);
  ImplicitRungeKutta stepper(rhs, Gauss3a, Gauss3b, Gauss3c);


  /*
  // arbitrary order Gauss-Legendre
  int stages = 5;
  Vector<> c(stages), b1(stages);
  GaussLegendre(c, b1);

  auto [a, b] = computeABfromC(c);
  ImplicitRungeKutta stepper(rhs, a, b, c);
  */

  /* 
  // arbitrary order Radau
  int stages = 5;
  Vector<> c(stages), b1(stages);
  GaussRadau(c, b1);

  auto [a, b] = computeABfromC(c);
  ImplicitRungeKutta stepper(rhs, a, b, c);
  */


  std::ofstream outfile ("output_test_ode.txt");
  std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  outfile << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  for (int i = 0; i < steps; i++)
  {
     stepper.doStep(tau, y);

     std::cout << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
     outfile << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
  }

  // //--------------------------------------------------------------------------
  // // 1. Implicit Gauss Method (using pre-tabulated Gauss3c)
  // //--------------------------------------------------------------------------
  // {
  //     auto [Gauss3a, Gauss3b] = computeABfromC(Gauss3c);
  //     ImplicitRungeKutta stepper_gauss(rhs, Gauss3a, Gauss3b, Gauss3c);

  //     Vector<> y = y_init; // Reset IC
  //     std::ofstream outfile("output_test_ode_gauss.txt");
  //     std::cout << "Testing the Gauss method" << std::endl;

  //     // Write Initial State
  //     std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  //     outfile << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  //     for (int i = 0; i < steps; i++)
  //     {
  //         stepper_gauss.DoStep(tau, y);
  //         double t = (i + 1) * tau;
  //         std::cout << t << "  " << y(0) << " " << y(1) << std::endl;
  //         outfile << t << "  " << y(0) << " " << y(1) << std::endl;
  //     }
  // }

  // //--------------------------------------------------------------------------
  // // 2. Implicit Gauss-Legendre (Arbitrary order, 5 stages)
  // //--------------------------------------------------------------------------
  // {
  //     int stages = 5;
  //     Vector<> c(stages), b1(stages);
  //     GaussLegendre(c, b1);

  //     auto [a, b] = computeABfromC(c);
  //     ImplicitRungeKutta stepper_gauss_legendre(rhs, a, b, c);

  //     Vector<> y = y_init; // Reset IC
  //     std::ofstream outfile("output_test_ode_gauss_legendre.txt");
  //     std::cout << "Testing the Gauss-Legendre method" << std::endl;

  //     std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  //     outfile << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  //     for (int i = 0; i < steps; i++)
  //     {
  //         stepper_gauss_legendre.DoStep(tau, y);
  //         double t = (i + 1) * tau;
  //         std::cout << t << "  " << y(0) << " " << y(1) << std::endl;
  //         outfile << t << "  " << y(0) << " " << y(1) << std::endl;
  //     }
  // }

  // //--------------------------------------------------------------------------
  // // 3. Implicit Gauss-Radau (Arbitrary order, 5 stages)
  // //--------------------------------------------------------------------------
  // {
  //     int stages = 5;
  //     Vector<> c(stages), b1(stages);
  //     GaussRadau(c, b1);

  //     auto [a, b] = computeABfromC(c);
  //     ImplicitRungeKutta stepper_gauss_radau(rhs, a, b, c);

  //     Vector<> y = y_init; // Reset IC
  //     std::ofstream outfile("output_test_ode_gauss_radau.txt");
  //     std::cout << "Testing the Gauss-Radau method" << std::endl;

  //     std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  //     outfile << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  //     for (int i = 0; i < steps; i++)
  //     {
  //         // Note: Check if method is named DoStep or doStep in your library headers
  //         stepper_gauss_radau.doStep(tau, y);
  //         double t = (i + 1) * tau;
  //         std::cout << t << "  " << y(0) << " " << y(1) << std::endl;
  //         outfile << t << "  " << y(0) << " " << y(1) << std::endl;
  //     }
  // }

  // //--------------------------------------------------------------------------
  // // 4. Explicit RK4
  // //--------------------------------------------------------------------------
  // {
  //     Matrix<> A(4, 4);
  //     A = 0.0;
  //     A(1, 0) = 0.5;
  //     A(2, 1) = 0.5;
  //     A(3, 2) = 1.0;

  //     Vector<> b{1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6};
  //     Vector<> c{0.0, 0.5, 0.5, 1.0};

  //     ExplicitRungeKutta stepper_rk4(rhs, A, b, c);

  //     Vector<> y = y_init; // Reset IC
  //     std::ofstream outfile("output_test_ode_rk4.txt");
  //     std::cout << "Testing explicit RK4\n";

  //     std::cout << 0.0 << "  " << y(0) << " " << y(1) << "\n";
  //     outfile << 0.0 << "  " << y(0) << " " << y(1) << "\n";

  //     for (int i = 0; i < steps; ++i)
  //     {
  //         stepper_rk4.DoStep(tau, y);
  //         double t = (i + 1) * tau;
  //         std::cout << t << "  " << y(0) << " " << y(1) << "\n";
  //         outfile << t << "  " << y(0) << " " << y(1) << "\n";
  //     }
  // }

  // //--------------------------------------------------------------------------
  // // 5. Explicit RK2 (Midpoint)
  // //--------------------------------------------------------------------------
  // {
  //     // Butcher tableau:
  //     // 0   |
  //     // 1/2 | 1/2
  //     // ----+---------
  //     //     | 0    1

  //     Matrix<> A(2, 2);
  //     A = 0.0;
  //     A(1, 0) = 0.5; // lower triangular

  //     Vector<> b{0.0, 1.0};
  //     Vector<> c{0.0, 0.5};

  //     ExplicitRungeKutta stepper_rk2(rhs, A, b, c);

  //     Vector<> y = y_init; // Reset IC
  //     std::ofstream outfile("output_test_ode_rk2.txt");
  //     std::cout << "Testing explicit RK2\n";

  //     std::cout << 0.0 << "  " << y(0) << " " << y(1) << "\n";
  //     outfile << 0.0 << "  " << y(0) << " " << y(1) << "\n";

  //     for (int i = 0; i < steps; ++i)
  //     {
  //         stepper_rk2.DoStep(tau, y);
  //         double t = (i + 1) * tau;
  //         std::cout << t << "  " << y(0) << " " << y(1) << "\n";
  //         outfile << t << "  " << y(0) << " " << y(1) << "\n";
  //     }
  // }

  // //--------------------------------------------------------------------------
  // // 6. Explicit Nystrom's Fifth Order Method
  // //--------------------------------------------------------------------------
  // {
  //     Matrix<> A(6, 6);
  //     A = 0.0;

  //     // row 2 (c = 1/3)
  //     A(1, 0) = 1.0 / 3.0;

  //     // row 3 (c = 2/5)
  //     A(2, 0) = 4.0 / 25.0;
  //     A(2, 1) = 6.0 / 25.0;

  //     // row 4 (c = 1)
  //     A(3, 0) = 1.0 / 4.0;
  //     A(3, 1) = -3.0;
  //     A(3, 2) = 15.0 / 4.0;

  //     // row 5 (c = 2/3)
  //     A(4, 0) = 2.0 / 27.0;
  //     A(4, 1) = 10.0 / 9.0;
  //     A(4, 2) = -50.0 / 81.0;
  //     A(4, 3) = 8.0 / 81.0;

  //     // row 6 (c = 4/5)
  //     A(5, 0) = 2.0 / 25.0;
  //     A(5, 1) = 12.0 / 25.0;
  //     A(5, 2) = 2.0 / 15.0;
  //     A(5, 3) = 8.0 / 75.0;

  //     // b vector (weights)
  //     Vector<> b(6);
  //     b(0) = 23.0 / 192.0;
  //     b(1) = 0.0;
  //     b(2) = 125.0 / 192.0;
  //     b(3) = 0.0;
  //     b(4) = -27.0 / 64.0;
  //     b(5) = 125.0 / 192.0;

  //     // c vector
  //     Vector<> c(6);
  //     c(0) = 0.0;
  //     c(1) = 1.0 / 3.0;
  //     c(2) = 2.0 / 5.0;
  //     c(3) = 1.0;
  //     c(4) = 2.0 / 3.0;
  //     c(5) = 4.0 / 5.0;

  //     ExplicitRungeKutta stepper_ny(rhs, A, b, c);

  //     Vector<> y = y_init; // Reset IC
  //     std::ofstream outfile("output_test_ode_nystrom.txt");
  //     std::cout << "Testing explicit RK Nystrom\n";

  //     std::cout << 0.0 << "  " << y(0) << " " << y(1) << "\n";
  //     outfile << 0.0 << "  " << y(0) << " " << y(1) << "\n";

  //     for (int i = 0; i < steps; ++i)
  //     {
  //         stepper_ny.DoStep(tau, y);
  //         double t = (i + 1) * tau;
  //         std::cout << t << "  " << y(0) << " " << y(1) << "\n";
  //         outfile << t << "  " << y(0) << " " << y(1) << "\n";
  //     }
  // }

  // return 0;
}