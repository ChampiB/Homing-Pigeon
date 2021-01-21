//
// Created by tmac3 on 06/01/2021.
//

#include <map>
#include <cmath>
#include "Functions.h"
#include "distributions/Distribution.h"
#include "distributions/Categorical.h"

using namespace Eigen;
using namespace hopi::distributions;
using namespace std;

namespace hopi::math {

    MatrixXd Functions::softmax(MatrixXd &vector) {
        MatrixXd res = vector.array() - vector.maxCoeff();
        res = res.array().exp();
        double sum = res.sum();

        if (sum == 0) {
            return MatrixXd::Constant(vector.rows(), 1, 1.0 / res.size());
        }
        return res / sum;
    }

    double Functions::KL(Distribution *d1, Distribution *d2) {
        static map<int, double (*)(Distribution *, Distribution *)> mapping{
                make_pair(DistributionType::CATEGORICAL, &Functions::KL_Categorical),
                make_pair(DistributionType::DIRICHLET, &Functions::KL_Dirichlet)
        };

        if (d1->type() != d2->type()) {
            throw runtime_error("Unsupported: KL between two distributions of different types.");
        }
        auto funcIt = mapping.find(d1->type());
        if (funcIt == mapping.end()) {
            throw runtime_error("Unsupported: KL does not support this type of distributions.");
        }
        return (*funcIt).second(d1, d2);
    }

    double Functions::KL_Categorical(Distribution *d1, Distribution *d2) {
        double kl = 0;
        auto c1 = dynamic_cast<Categorical*>(d1);
        auto c2 = dynamic_cast<Categorical*>(d2);

        for (int i = 0; i < c1->cardinality(); ++i) {
            kl += c1->p(i) * ( log(c1->p(i)) - log(c2->p(i)) );
        }
        return kl;
    }

    double Functions::KL_Dirichlet(Distribution *d1, Distribution *d2) {
        double kl = 0;
        auto p1 = d1->params();
        auto p2 = d2->params();

        for (int i = 0; i < p1.size(); ++i) {
            for (int j = 0; j < p1[i].cols(); ++j) {
                double sum1, sum2 = 0;
                for (int k = 0; k < p1[i].rows(); ++k) {
                    sum1 = p1[i](k, j);
                    sum2 = p2[i](k, j);
                }
                kl += log(tgamma(sum1)) - log(tgamma(sum2));
                for (int k = 0; k < p1[i].rows(); ++k) {
                    kl += log(tgamma(p2[i](k, j))) \
                        - log(tgamma(p1[i](k, j))) \
                        + (p1[i](k, j) - p2[i](k, j)) * (digamma(p1[i](k, j)) - digamma(sum1));
                }
            }
        }
        return kl;
    }

    double Functions::beta(MatrixXd x) {
        double res = 1;

        for (int i = 0; i < x.rows(); ++i) {
            res *= tgamma(x(i));
        }
        return res / tgamma(x.sum());
    }

    double Functions::log_beta(MatrixXd x)
    {
        double res = 0;

        for (int i = 0; i < x.rows(); ++i) {
            res += lgamma(x(i));
        }
        return res - lgamma(x.sum());
    }

    double Functions::digamma(double x) {
        static double c = 8.5;
        static double euler_mascheroni = 0.57721566490153286060;
        double r;
        double value;
        double x2;

        if (x <= 0.0)
            throw logic_error("Can't compute digamma function for negative value of x.");

        //  Use approximation for small argument.
        if (x <= 0.000001) {
            value = -euler_mascheroni - 1.0f / x + 1.6449340668482264365f * x;
            return value;
        }

        //  Reduce to DIGAMA(X + N).
        value = 0.0;
        x2 = x;
        while (x2 < c) {
            value -= 1.0f / x2;
            x2 += 1.0;
        }

        //  Use Stirling's (actually de Moivre's) expansion.
        r = 1.0f / x2;
        value += log(x2) - 0.5f * r;
        r = r * r;
        value -= r * (1.0f / 12.0f
                      - r * (1.0f / 120.0f
                             - r * (1.0f / 252.0f
                                    - r * (1.0f / 240.0f
                                           - r * (1.0f / 132.0f)))));

        return value;
    }

}