//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_ENVIRONMENT_H
#define HOMING_PIGEON_2_ENVIRONMENT_H

#include <Eigen/Dense>

namespace hopi::environments {

    class Environment {
    public:
        virtual int execute(int action) = 0;
        virtual void print() const = 0;
        [[nodiscard]] virtual int actions() const = 0;
        [[nodiscard]] virtual int states() const = 0;
        [[nodiscard]] virtual int observations() const = 0;
        virtual Eigen::MatrixXd A() = 0;
        virtual std::vector<Eigen::MatrixXd> B() = 0;
        virtual Eigen::MatrixXd D() = 0;
    };

}

#endif //HOMING_PIGEON_2_ENVIRONMENT_H
