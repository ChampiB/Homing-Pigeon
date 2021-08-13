//
// Created by Theophile Champion on 06/01/2021.
//

#include <map>
#include <random>
#include <cmath>
#include "Ops.h"
#include "distributions/Distribution.h"
#include "api/API.h"

using namespace torch;
using namespace torch::indexing;
using namespace hopi::distributions;
using namespace hopi::api;
using namespace std;

namespace hopi::math {

    double Ops::kl(Distribution *d1, Distribution *d2) {
        static map<int, double (*)(Distribution *, Distribution *)> mapping{
                {DistributionType::CATEGORICAL, &Ops::kl_categorical},
                {DistributionType::DIRICHLET, &Ops::kl_dirichlet}
        };

        assert(d1->type() == d2->type() && "Ops::kl, both input distributions must have the same type.");
        auto funcIt = mapping.find(d1->type());
        assert(funcIt != mapping.end() && "Ops::kl, unsupported distribution type.");
        return (*funcIt).second(d1, d2);
    }

    double Ops::kl_categorical(Distribution *d1, Distribution *d2) {
        return (d1->params() * (d1->logParams() - d2->logParams())).sum().item<double>();
    }

    double Ops::kl_dirichlet(const Tensor &t1, const Tensor &t2) {
        double kl = 0;
        auto t1_a = t1.accessor<double,3>();
        auto t2_a = t2.accessor<double,3>();

        for (int i = 0; i < t1_a.size(0); ++i) {
            for (int j = 0; j < t1_a.size(1); ++j) {
                auto sum1 = t1.index({i,j,Ellipsis}).sum().item<double>();
                auto sum2 = t2.index({i,j,Ellipsis}).sum().item<double>();
                kl += log_gamma(sum1) - log_gamma(sum2);
                for (int k = 0; k < t1_a.size(2); ++k) {
                    kl += log_gamma(t2_a[i][j][k]) \
                        - log_gamma(t1_a[i][j][k]) \
                        + (t1_a[i][j][k] - t2_a[i][j][k]) * (digamma(t1_a[i][j][k]) - digamma(sum1));
                }
            }
        }
        return kl;
    }

    double Ops::kl_dirichlet(Distribution *d1, Distribution *d2) {
        auto p1 = d1->params();
        auto p2 = d2->params();

        assert(p1.dim() == p2.dim() && "Ops::kl_dirichlet, input must have the same dimension.");
        assert(p1.sizes() == p2.sizes() && "Ops::kl_dirichlet, input must have the same sizes.");
        Ops::unsqueeze(3 - p1.dim(), {&p1, &p2});
        return kl_dirichlet(p1, p2);
    }

    double Ops::beta(const Tensor &x) {
        auto x_a = x.accessor<double,1>();
        double res = 1;
        double sum = 0;

        for (int i = 0; i < x_a.size(0); ++i) {
            res *= tgamma(x_a[i]);
            sum += x_a[i];
        }
        return res / tgamma(sum);
    }

    double Ops::log_beta(const Tensor &x) {
        return log(beta(x));
    }

    double Ops::log_gamma(double x) {
        return log(tgamma(x));
    }

    double Ops::digamma(double x) {
        static double c = 8.5;
        static double euler_mascheroni = 0.57721566490153286060;
        double r;
        double value;
        double x2;

        assert(x > 0.0 && "Ops::digamma, input must be strictly positive.");

        //  Use approximation for small argument.
        if (x <= 0.000001) {
            value = -euler_mascheroni - 1.0f / x + 1.6449340668482264365f * x;
            return value;
        }

        //  Reduce to digamma(X + N).
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

    Tensor Ops::one_hot(int size, int index) {
        Tensor vec = API::zeros({size});

        vec[index] = 1;
        return vec.to(kDouble);
    }

    Tensor Ops::uniform(const IntArrayRef &sizes, int dim) {
        // Check that inputs are correct
        size_t dims = sizes.size();
        assert(dims <= 3 && "Ops::uniform, dimension described by input sizes must be <= 3.");
        assert(dim < dims && "Ops::uniform, input \"dim\" must be inferior to the number of dimensions.");

        // Implement the function
        auto n_elems = (double) sizes[dim];

        return API::full(sizes, 1.0 / n_elems);
    }

    void Ops::unsqueeze(long n, std::initializer_list<Tensor *> tensors) {
        for (int i = 0; i < n; ++i) {
            for (auto j : tensors) {
                *j = torch::unsqueeze(*j, 0);
            }
        }
    }

    Tensor Ops::multiplication(const Tensor &x1, const Tensor &x2, std::initializer_list<int> ml) {
        // Create the list on non-matching dimensions
        std::vector<int64_t> not_ml;

        for (int64_t i = 0; i < x1.dim(); ++i) {
            if (std::find(ml.begin(), ml.end(), i) == ml.end()) {
                not_ml.push_back(i);
            }
        }

        // Sequence of expansions
        Tensor x2_tmp = x2;
        for (int64_t i : not_ml) {
            x2_tmp = expansion(x2_tmp, x1.size(i), x2_tmp.dim());
        }

        // Permutation
        std::vector<int64_t> pl(x1.dim());
        for (int i = 0; i < x1.dim(); ++i) {
            auto it = std::find(ml.begin(), ml.end(), i);
            if (it != ml.end()) { // if "i" is in matching list
                pl[i] = it - ml.begin();
            } else {
                pl[i] = (int64_t) ml.size() + (std::find(not_ml.begin(), not_ml.end(), i) - not_ml.begin());
            }
        }
        x2_tmp = x2_tmp.permute(IntArrayRef(pl));

        // Element-wise multiplication
        return x2_tmp * x1;
    }

    Tensor Ops::average(
            const Tensor &x1, const Tensor &x2,
            std::initializer_list<int> ml, std::initializer_list<int> el
    ) {
        // Perform the element-wise multiplication
        Tensor result = multiplication(x1, x2, ml);

        // Create the reduction list, i.e. rl = ml \ el where "\" = set minus
        std::vector<int> rl(ml);

        rl.erase(std::remove_if(rl.begin(), rl.end(), [el](const int &elem){
            return std::find(el.begin(), el.end(), elem) != el.end();
        }), rl.end());

        // Sort the reduction list in decreasing order
        std::sort(rl.begin(), rl.end());
        std::reverse(rl.begin(), rl.end());

        // Reduction of the tensor (using a summation) along the dimension of the reduction list
        for (int i : rl) {
            result = result.sum(i);
        }
        return result;
    }

    Tensor Ops::expansion(const Tensor &x1, long n, long dim) {
        Tensor result = torch::unsqueeze(x1, dim);
        std::vector<long> sizes(result.dim());

        for (int i = 0; i < result.dim(); ++i) {
            sizes[i] = (i != dim) ? -1 : n;
        }
        return result.expand(IntArrayRef(sizes));
    }

    Tensor Ops::outer_tensor_product(std::initializer_list<Tensor *> ts) {
        assert(ts.size() > 0 && "Ops::outer_tensor_product, input list must contains at least one element");
        Tensor result = **ts.begin();

        for (auto t = ts.begin() + 1; t != ts.end(); ++t) {
            assert((*t)->dim() == 1 && "Ops::outer_tensor_product, each tensor in the input list must be a vector");
            Tensor tmp = **t;
            for (int i = 0; i < t - ts.begin(); ++i) {
                Ops::expansion(**t, result.size(i), i);
            }
            result = Ops::expansion(result, (*t)->size((*t)->dim() - 1), result.dim()) * tmp;
        }
        return result;
    }

    int Ops::randomInt(const torch::Tensor &w) {
        return randomInt(API::toStdVector(w));
    }

    int Ops::randomInt(int max) {
        static std::random_device dev;
        static std::mt19937 engine(dev());
        std::uniform_int_distribution<int> rand_int(0, max);

        return rand_int(engine);
    }

    int Ops::randomInt(const std::vector<double> &weights) {
        static std::random_device dev;
        static std::mt19937 engine(dev());
        std::discrete_distribution<int> rand_int(weights.begin(), weights.end());

        return rand_int(engine);
    }

}