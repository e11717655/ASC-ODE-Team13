#ifndef LEGENDRE_HPP
#define LEGENDRE_HPP

#include <vector>

template <typename T>
void LegendrePolynomials(int n, T x, std::vector<T>& P) {
    if (n < 0) { P.clear(); return; }

    P.resize(n + 1);
    P[0] = T(1);
    if (n == 0) return;

    P[1] = x;

    for (int k = 2; k <= n; ++k) {
        P[k] = ((T(2*k - 1) * x * P[k - 1]) -
                (T(k - 1) * P[k - 2])) / T(k);
    }
}

#endif