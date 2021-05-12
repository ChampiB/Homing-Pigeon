//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_ALGOVMP_H
#define HOMING_PIGEON_2_ALGOVMP_H

#include <memory>
#include <vector>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::algorithms {

    class AlgoVMP {
    public:
        static void inference(const std::vector<nodes::VarNode*>& vars, double epsilon = 0.01, int max_iter = 2147483647);
        static void inference(nodes::VarNode *var);
        static double vfe(const std::vector<nodes::VarNode*>& vars);
    };

}

#endif //HOMING_PIGEON_2_ALGOVMP_H
