//
// Created by Theophile Champion on 10/05/2021.
//

#ifndef HOMINGPIGEON_API_H
#define HOMINGPIGEON_API_H

#include <torch/torch.h>
#include "Aliases.h"

namespace hopi::api {

    /**
     * This class contains the "Application Programming Interface" through which a user can use the framework.
     */
    class API {
    public:
        //
        // The following functions create random variables distributed according to various distributions.
        // It also handle the creation of the underlying factor graph that will be used to perform inference.
        //

        /**
         * Create a categorical random variable whose distribution is defined by the parameters "param".
         * @param param the parameters of the distribution
         * @return the random variable
         */
        static RV *Categorical(const torch::Tensor& param);

        /**
         * Create a categorical random variable whose distribution is defined by the parameters "param". Note that
         * "param" is random variable which is assumed to be distributed according to a Dirichlet distribution.
         * @param param the (Dirichlet) random variable representing the parameters of the distribution
         * @return the (Categorical) random variable
         */
        static RV *Categorical(RV *param);

        /**
         * Create a transition random variable whose distribution is defined by the parameters "param".
         * This (Transition) random variable is condition on another random variable "s", i.e., P(o|s) where the
         * created random variable is called "o".
         * @param s the random variable on which the created random variable is conditioned
         * @param param the parameters of the transition distribution
         * @return the (Transition) random variable
         */
        static RV *Transition(RV *s, const torch::Tensor& param);

        /**
         * Create a transition random variable whose distribution is defined by the random variable "param",
         * which is assumed to be distributed according to a Dirichlet distribution. This (Transition) random variable
         * is condition on another random variable "s", i.e., P(o|s,A) where "o" is the random variable return by this
         * function and "A" is a matrix containing the distribution's parameters.
         * @param s the random variable on which the created random variable is conditioned
         * @param param the parameters of the transition distribution
         * @return the (Transition) random variable
         */
        static RV *Transition(RV *s, RV *param);

        /**
         * Create a random variable distributed according to an active transition distribution,
         * i.e., P(s_next|s,a) where "s_next" is the returned random variable, "a" is the action upon which the
         * transition is conditioned and "s" is the state upon which the transition is conditioned.
         * @param s the state upon which the transition is conditioned
         * @param a the action upon which the transition is conditioned
         * @param param the parameters of the ActiveTransition distribution
         * @return the created ransom variable
         */
        static RV *ActiveTransition(RV *s, RV *a, const torch::Tensor& param);

        /**
         * Create a random variable distributed according to an active transition, i.e., P(s_next|s,a,B) where "s_next"
         * is the returned random variable, "a" the action upon which the transition is conditioned and "s" is the
         * state upon which the transition is conditioned and "B" is a 3d tensor containing the parameters of the
         * distribution. Note that this 3d tensor is assumed to be distributed according to a Dirichlet distribution.
         * @param s the state upon which the transition is conditioned
         * @param a the action upon which the transition is conditioned
         * @param param the parameters of the ActiveTransition distribution
         * @return the created ransom variable
         */
        static RV *ActiveTransition(RV *s, RV *a, RV *param);

        /**
         * Create a random variable distributed according to a Dirichlet distribution with parameters "param".
         * @param param the parameters of the Dirichlet distribution
         * @return the created random variable
         */
        static RV *Dirichlet(const torch::Tensor& param);

        //
        // The following functions handle the creation of tensor.
        //

        /**
         * Setter.
         * @param type the type of data that must be used when creating a tensor
         */
        static void setDataType(const at::ScalarType& type);

        /**
         * Getter.
         * @return the type of data to use when creating a tensor
         */
        static at::ScalarType dataType();

        /**
         * Create a tensor filled with uninitialized values.
         * @param sizes the sizes of the dimensions of the tensor that must be created
         * @return the empty tensor
         */
        static torch::Tensor empty(at::IntArrayRef sizes);

        /**
         * Create a tensor from a specific list of values.
         * @param data the value composing the tensor to be created
         * @return the created tensor
         */
        static torch::Tensor tensor(const torch::detail::TensorDataContainer &&data);

        /**
         * Create a tensor full of zeros.
         * @param sizes the sizes of the dimensions of the tensor to create
         * @return the created tensor
         */
        static torch::Tensor zeros(at::IntArrayRef sizes);

        /**
         * Create a tensor filled with a specific value
         * @tparam T the value type
         * @param sizes the sizes of the dimensions of the tensor to be created
         * @param value the value to fill the tensor with
         * @return the created tensor
         */
        template<class T> static torch::Tensor full(at::IntArrayRef sizes, T value) {
            return torch::full(sizes, value).to(dataType());
        }
    };

}


#endif //HOMINGPIGEON_API_H
