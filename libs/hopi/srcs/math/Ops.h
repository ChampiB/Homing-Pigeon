//
// Created by Theophile Champion on 06/01/2021.
//

#ifndef EXPERIMENTS_AI_TS_FUNCTIONS_H
#define EXPERIMENTS_AI_TS_FUNCTIONS_H

#include <torch/torch.h>

namespace hopi::distributions {
    class Distribution;
}

namespace hopi::math {

    /**
     * This class contains mathematical functions and operators used throughout the framework.
     */
    class Ops {
    public:
        //
        // Math functions
        //

        /**
         * Compute the Kullback-Leibler (KL) divergence between the two distributions
         * @param d1 first distribution
         * @param d2 second distribution
         * @return the kL divergence
         */
        static double kl(distributions::Distribution *d1, distributions::Distribution *d2);

        /**
         * Compute the Kullback-Leibler (KL) divergence between the two categorical distributions
         * @param d1 first categorical distribution
         * @param d2 second categorical distribution
         * @return the kL divergence
         */
        static double kl_categorical(distributions::Distribution *d1, distributions::Distribution *d2);

        /**
         * Compute the Kullback-Leibler (KL) divergence between the two Dirichlet distributions
         * @param d1 first Dirichlet distribution
         * @param d2 second Dirichlet distribution
         * @return the kL divergence
         */
        static double kl_dirichlet(distributions::Distribution *d1, distributions::Distribution *d2);

        /**
         * Compute the logarithm of the generalised beta function.
         * @param x the function's inputs
         * @return the logarithm of the generalised beta function of x
         */
        static double log_beta(const torch::Tensor &x);

        /**
         * Compute the logarithm of the gamma function.
         * @param x the function's inputs
         * @return the logarithm of the gamma function of x
         */
        static double log_gamma(double x);

        /**
         * Compute the beta function.
         * @param x the function's inputs
         * @return the beta function of x
         */
        static double beta(const torch::Tensor &x);

        /**
         * Compute the digamma function of x
         * @param x the input
         * @return the digamma function of x
         */
        static double digamma(double x);

        //
        // Matrices creation
        //

        /**
         * Create a one-hot vector
         * @param size the vector's size
         * @param index the index at which a one should be placed
         * @return the one-hot vector
         */
        static torch::Tensor one_hot(int size, int index);

        /**
         * Create a tensor in which the elements are equal to one over the number of elements in the dimension "dim",
         * e.g., uniform({2,3}, 0) will return the following matrix:
         *    | 0.5 0.5 0.5 |
         *    | 0.5 0.5 0.5 |
         * where each element equals 0.5 = 1 / number of rows.
         * @param sizes the sizes of each dimension of the tensor being created
         * @param dim the dimension along which the tensor must store uniform distributions
         * @return the created tensor
         */
        static torch::Tensor uniform(const torch::IntArrayRef &sizes, int dim = 0);

        //
        // Tensor operators (ml = matching list, el = exclusion list)
        //

        /**
         * Unsqueeze oll tensors sent as parameters n times. Note that the dimension is added before all other
         * dimensions, i.e., at the end.
         * @param n the number of times each tensor must be unsqueezed
         * @param tensors the tensors to be unsqueezed
         */
        static void unsqueeze(long n, std::initializer_list<torch::Tensor *> tensors);

        /**
         * Performs an expansion of the tensor "x1", this expansion will lead to a new dimension of size "n"
         * inserted after the dimension "dim". More information about the expansion operator can be found in
         * the paper named: "The compelling free energy: A novel perspective on active inference".
         * @param x1 the tensor to be expanded
         * @param n the size of the new dimension
         * @param dim the dimension after which the expansion should be performed
         * @return the expended tensor
         */
        static torch::Tensor expansion(const torch::Tensor &x1, long n, long dim);

        /**
         * Compute the element-wise multiplication of "x1" by "x2" after properly expanding "x2" such that
         * its shape matches the shape of "x1".
         *
         * Details: This operator performs tensor expansions so that the sizes of of the dimensions of "x2" matches
         * the size of the dimensions of "x1". Then, it re-arranges the dimensions of "x2" by performing a permutation
         * of its dimensions so that the shape of "x2" matches the shape of "x1". Note that the permutation uses the
         * matching list "ml" to know which dimensions of "x2" should be matched to which dimensions of "x1".
         *
         * More details can be found in the paper named:
         *     "The compelling free energy: A novel perspective on active inference".
         *
         * @param x1 the first tensor
         * @param x2 the second tensor
         * @param ml the matching list
         * @return the element-wise multiplication of "x1" by "x2" after properly expanding "x2"
         */
        static torch::Tensor multiplication(
                const torch::Tensor &x1, const torch::Tensor &x2, std::initializer_list<int> ml
        );

        /**
         * Compute the weighted average of "x1" along the dimension specified by the matching list, and the weights
         * of the average are the elements of "x2".
         *
         * Details: The average operator first perform a element-wise multiplication by matching the dimensions of
         * "x2" to the dimensions of "x1" using the matching list "ml". Then, it performs a summation over the
         * dimensions of "x2" that does not belong to the exclusion list "el".
         *
         * More details can be found in the paper named:
         *     "The compelling free energy: A novel perspective on active inference".
         *
         * @param x1 the first tensor
         * @param x2 the second tensor
         * @param ml the matching list
         * @param el the exclusion list
         * @return the average of "x1" with respect to "x2" according to the matching and exclusion list
         */
        static torch::Tensor average(
                const torch::Tensor &x1, const torch::Tensor &x2,
                std::initializer_list<int> ml, std::initializer_list<int> el = {}
        );

        /**
         * Compute the outer tensor product between the tensors sent as parameters.
         * @param ts the input tensors
         * @return the outer tensor product
         */
        static torch::Tensor outer_tensor_product(std::initializer_list<torch::Tensor *> ts);

    private:
        //
        // Functions that the final user should never call directly
        //

        /**
         * Implement the "hard logic" of the KL divergence between two Dirichlet distributions,
         * i.e., assuming well formatted inpute.
         * @param t1 the first 3-tensor of parameters
         * @param t2 the second 3-tensor of parameters
         * @return the KL divergence
         */
        static double kl_dirichlet(const torch::Tensor &t1, const torch::Tensor &t2);

    };

}

#endif //EXPERIMENTS_AI_TS_FUNCTIONS_H
