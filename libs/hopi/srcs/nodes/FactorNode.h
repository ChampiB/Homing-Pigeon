//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_FACTORNODE_H
#define HOMING_PIGEON_2_FACTORNODE_H

#include <Eigen/Dense>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::nodes {

    class FactorNode {
    public:
        virtual VarNode *parent(int index) = 0;
        virtual VarNode *child() = 0;
        virtual std::vector<Eigen::MatrixXd> message(VarNode *to) = 0;
        [[nodiscard]] std::string name() const;
        void setName(std::string name);
        virtual double vfe() = 0;

    private:
        std::string _name;
    };

}

#endif //HOMING_PIGEON_2_FACTORNODE_H
