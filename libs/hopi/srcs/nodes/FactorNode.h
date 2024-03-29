//
// Created by Theophile Champion on 28/11/2020.
//

#ifndef HOMING_PIGEON_FACTOR_NODE_H
#define HOMING_PIGEON_FACTOR_NODE_H

#include <torch/torch.h>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::nodes {

    /**
     * Interface representing a general factor node.
     */
    class FactorNode {
    public:
        /**
         * Getter.
         * @param i the index of the parent that must be returned
         * @return the i-the parent of the factor
         */
        virtual VarNode *parent(int i) = 0;

        /**
         * Getter.
         * @return the child of the factor, i.e., the random variable generated by the factor.
         */
        virtual VarNode *child() const = 0;

        /**
         * Compute the message towards a specific node
         * @param to the node toward which the message is sent
         * @return the message
         */
        virtual torch::Tensor message(VarNode *to) = 0;

        /**
         * Compute the Variational Free Energy (VFE) of the factor
         * @return the VFE
         */
        virtual double vfe() = 0;

        /**
         * Getter.
         * @return the name of the factor
         */
        [[nodiscard]] std::string name() const;

        /**
         * Setter.
         * @param name the new new of the factor
         */
        void setName(std::string &name);

        /**
         * Setter.
         * @param name the new new of the factor
         */
        void setName(std::string &&name);

    private:
        std::string _name;
    };

}

#endif //HOMING_PIGEON_FACTOR_NODE_H
