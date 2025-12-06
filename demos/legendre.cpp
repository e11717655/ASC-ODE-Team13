#include <iostream>
#include <fstream>
#include <vector>
#include "autodiff.hpp"
#include "legendre.hpp"

using ASC_ode::AutoDiff;

int main() {
    const int N = 5;

    std::ofstream file("../results/legendre_output.csv");
    file << "x";
    for (int i = 0; i <= N; i++) file << ",P" << i << ",dP" << i;
    file << "\n";

    for (double t = -1.0; t <= 1.0; t += 0.01) {

        AutoDiff<1,double> x(t);
        x.deriv()[0] = 1.0;

        std::vector<AutoDiff<1,double>> P;
        LegendrePolynomials(N, x, P);

        file << t;
        for (int i = 0; i <= N; i++)
            file << "," << P[i].value() << "," << P[i].deriv()[0];
        file << "\n";
    }

    std::cout << "Wrote legendre_output.csv (polynomials + derivatives)\n";
    return 0;
}
