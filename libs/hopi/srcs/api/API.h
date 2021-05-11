//
// Created by tmac3 on 10/05/2021.
//

#ifndef HOMINGPIGEON_API_H
#define HOMINGPIGEON_API_H

#include "Eigen/Dense"
#include "Aliases.h"

namespace hopi::api {

    class API {
    public:
        /*
         * The following function create random variables distributed according to various distributions.
         * It also handle the creation of the underlying factor graph that will be used to perform inference.
         */
        static RV *Categorical(const Eigen::MatrixXd& param);
        static RV *Categorical(RV *param);
        static RV *Transition(RV *s, const Eigen::MatrixXd& param);
        static RV *Transition(RV *s, RV *param);
        static RV *ActiveTransition(RV *s, RV *a, const std::vector<Eigen::MatrixXd>& param);
        static RV *ActiveTransition(RV *s, RV *a, RV *param);
        static RV *Dirichlet(const std::vector<Eigen::MatrixXd>& param);
        static RV *Dirichlet(const Eigen::MatrixXd& param);
    };

};


#endif //HOMINGPIGEON_API_H
