//
// Created by tmac3 on 10/05/2021.
//

#ifndef HOMINGPIGEON_API_H
#define HOMINGPIGEON_API_H

#include <torch/torch.h>
#include "Aliases.h"

namespace hopi::api {

    class API {
    public:
        /*
         * The following function create random variables distributed according to various distributions.
         * It also handle the creation of the underlying factor graph that will be used to perform inference.
         */
        static RV *Categorical(const torch::Tensor& param);
        static RV *Categorical(RV *param);
        static RV *Transition(RV *s, const torch::Tensor& param);
        static RV *Transition(RV *s, RV *param);
        static RV *ActiveTransition(RV *s, RV *a, const torch::Tensor& param);
        static RV *ActiveTransition(RV *s, RV *a, RV *param);
        static RV *Dirichlet(const torch::Tensor& param);
    };

};


#endif //HOMINGPIGEON_API_H
