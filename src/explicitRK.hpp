#ifndef EXPLICITRK_HPP
#define EXPLICITRK_HPP

#include <vector.hpp>
#include <matrix.hpp>
#include <inverse.hpp>

namespace ASC_ode
{
    using namespace nanoblas;

    class ExplicitRungeKutta : public TimeStepper
    {
        Matrix<> m_a;
        Vector<> m_b, m_c;
        int m_stages;
        int m_n;
        Vector<> m_k;
        Vector<> m_ytmp;

    public:
        ExplicitRungeKutta(std::shared_ptr<NonlinearFunction> rhs,
                           const Matrix<> &a, const Vector<> &b, const Vector<> &c)
            : TimeStepper(rhs), m_a(a), m_b(b), m_c(c),
              m_stages(c.size()), m_n(rhs->dimX()),
              m_k(m_stages * m_n), m_ytmp(m_n)
        {
        }
        void DoStep(double tau, VectorView<double> y) override
        {
            for (size_t i = 0; i < m_stages; i++)
            {
                m_ytmp = y;

                for (size_t j = 0; j < i; j++)
                {
                    if (m_a(i, j) != 0.0)
                    {
                        m_ytmp += (tau * m_a(i, j)) * m_k.range(j * m_n, (j + 1) * m_n);
                    }
                }
                m_rhs->evaluate(m_ytmp, m_k.range(i * m_n, (i + 1) * m_n));
            }
            for (size_t j = 0; j < m_stages; j++)
            {
                if (m_b(j) != 0.0)
                {
                    y += (tau * m_b(j)) * m_k.range(j * m_n, (j + 1) * m_n);
                }
            }
        }
    };

}

#endif