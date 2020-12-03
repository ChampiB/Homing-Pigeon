//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_ALGOVMP_H
#define HOMING_PIGEON_2_ALGOVMP_H

#include <memory>
#include <vector>
#include <Eigen/Dense>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::algorithms {

    class AlgoVMP {
    public:
        static void inference(const std::vector<nodes::VarNode*>& vars, double epsilon = 0.01);
        static void inference(nodes::VarNode *var);
        static double vfe(const std::vector<nodes::VarNode*>& vars);
        static Eigen::MatrixXd softmax(Eigen::MatrixXd& vector);
    };

}

#endif //HOMING_PIGEON_2_ALGOVMP_H
