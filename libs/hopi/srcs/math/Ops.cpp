//
// Created by tmac3 on 06/01/2021.
//

#include <map>
#include <cmath>
#include "Ops.h"
#include "distributions/Distribution.h"

using namespace torch;
using namespace hopi::distributions;
using namespace std;

namespace hopi::math {

    double Ops::KL(Distribution *d1, Distribution *d2) {
        static map<int, double (*)(Distribution *, Distribution *)> mapping{
                {DistributionType::CATEGORICAL, &Ops::KL_Categorical},
                {DistributionType::DIRICHLET, &Ops::KL_Dirichlet}
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

    double Ops::KL_Categorical(Distribution *d1, Distribution *d2) {
        return (d1->params() * (d1->logParams() - d2->logParams())).sum().item<double>();
    }

    double Ops::KL_Dirichlet(Distribution *d1, Distribution *d2) {
        double kl = 0;
        auto p1 = d1->params();
        auto p2 = d2->params();
        auto sum1 = p1.sum(std::vector<int64_t>{1});
        auto sum2 = p2.sum(std::vector<int64_t>{1});

        for (int i = 0; i < p1.size(0); ++i) {
            for (int j = 0; j < p1.size(2); ++j) {
                kl += log(tgamma(sum1[i][j].item<double>())) - log(tgamma(sum2[i][j].item<double>()));
                for (int k = 0; k < p1.size(1); ++k) {
                    auto diff = (p1[i][k][j] - p2[i][k][j]).item<double>();
                    kl += log(tgamma(p2[i][k][j].item<double>())) - log(tgamma(p1[i][k][j].item<double>())) \
                        + diff * (digamma(p1[i][k][j].item<double>()) - digamma(sum1[i][j].item<double>()));
                }
            }
        }
        return kl;
    }

    double Ops::beta(const Tensor &x) {
        double res = 1;

        for (int i = 0; i < x.numel(); ++i) {
            res *= tgamma(x[i].item<double>());
        }
        return res / tgamma(x.sum().item<double>());
    }

    double Ops::log_beta(const Tensor &x)
    {
        double res = 0;

        for (int i = 0; i < x.numel(); ++i) {
            res += lgamma(x[i].item<double>());
        }
        return res - lgamma(x.sum().item<double>());
    }

    double Ops::digamma(double x) {
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

    Tensor Ops::oneHot(int size, int index) {
        Tensor vec = torch::full({size}, 0);

        vec[index] = 1;
        return vec;
    }

    Tensor Ops::uniformColumnWise(const torch::ArrayRef<long> &sizes) {
        size_t size = sizes.size();
        auto rows = (double)( (size == 1) ? sizes[0] : sizes[size - 2] );

        if (size > 3) {
            throw std::runtime_error("Unsupported call to uniformColumnWise with more than three dimensions");
        }
        return torch::full(sizes, 1.0 / rows);
    }

}