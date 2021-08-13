//
// Created by Theophile Champion on 28/11/2020.
//

#ifndef HOMING_PIGEON_VAR_NODE_H
#define HOMING_PIGEON_VAR_NODE_H

#include "VarNodeType.h"
#include <memory>
#include <vector>
#include <string>

namespace hopi::distributions {
    class Distribution;
}
namespace hopi::nodes {
    class FactorNode;
}
namespace hopi::algorithms::planning {
    class MCTSNodeData;
}

namespace hopi::nodes {

    /**
     * Class representing a variable node.
     */
    class VarNode {
    public:
        /**
         * Create a variable node.
         * @param type the type of variable the node represents, i.e., observed or hidden
         * @return the created node
         */
        static std::unique_ptr<VarNode> create(VarNodeType type);

    public:
        /**
         * Construct a variable node.
         * @param type the type of variable the node represents, i.e., observed or hidden
         */
        explicit VarNode(VarNodeType type);

        /**
         * Destruct the variable node.
         */
        ~VarNode();

        /**
         * Setter.
         * @param prior the new prior distribution
         */
        void setPrior(std::unique_ptr<distributions::Distribution> prior);

        /**
         * Setter.
         * @param posterior the new posterior distribution
         */
        void setPosterior(std::unique_ptr<distributions::Distribution> posterior);

        /**
         * Setter.
         * @param biased the new biased distribution
         */
        void setBiased(std::unique_ptr<distributions::Distribution> biased);

        /**
         * Setter.
         * @param parent the new parent factor node
         */
        void setParent(FactorNode *parent);

        /**
         * Setter.
         * @param type the new type of the node
         */
        void setType(VarNodeType type);

        /**
         * Setter.
         * @param name the new name of the node
         */
        void setName(std::string &name);

        /**
         * Setter.
         * @param name the new name of the node
         */
        void setName(std::string &&name);

        /**
         * Getter.
         * @return the node's parent factor
         */
        FactorNode *parent() const;

        /**
         * An iterator on the first child of the node
         * @return the iterator
         */
        std::vector<FactorNode *>::iterator firstChild();

        /**
         * An iterator on the last child of the node
         * @return the iterator
         */
        std::vector<FactorNode *>::iterator lastChild();

        /**
         * Add a child to the node
         * @param child the child that must be added
         */
        void addChild(FactorNode *child);

        /**
         * Getter.
         * @return the number of children the node has
         */
        [[nodiscard]] int nChildren() const;

        /**
         * Getter.
         * @return the number of children the node has
         */
        [[nodiscard]] int nChildrenHiddenStates() const;

        /**
         * Getter.
         * @return the number of children the node has
         */
        [[nodiscard]] VarNode *child(int id) const;

        /**
         * Getter.
         * @return the type of random variable this node represents
         */
        [[nodiscard]] VarNodeType type() const;

        /**
         * Getter.
         * @return the node's name
         */
        [[nodiscard]] std::string name() const;

        /**
         * Getter.
         * @return the node's data
         */
        [[nodiscard]] algorithms::planning::MCTSNodeData *data() const;

        /**
         * Getter.
         * @return the node's prior distribution
         */
        distributions::Distribution *prior() const;

        /**
         * Getter.
         * @return the node's posterior distribution
         */
        distributions::Distribution *posterior() const;

        /**
         * Getter.
         * @return the node's biased distribution
         */
        distributions::Distribution *biased() const;

        /**
         * In the vector of children, remove the pointers that are equals to nullptr.
         */
        void removeNullChildren();

        /**
         * Disconnect the child node sent as parameter from the variable node (i.e., from this). "Disconnect" means
         * that its pointer in the vector of children is set to nullptr.
         * @param node the child node that must be disconnected
         */
        void disconnectChild(nodes::FactorNode *node);

    private:
        std::unique_ptr<algorithms::planning::MCTSNodeData> _data;
        std::vector<FactorNode *> _children;
        FactorNode *_parent;
        std::string _name;
        std::unique_ptr<distributions::Distribution> _prior;
        std::unique_ptr<distributions::Distribution> _posterior; // Evidence for OBSERVED variables
        std::unique_ptr<distributions::Distribution> _biased;    // Prior preferences
        VarNodeType _type;
    };

}

#endif //HOMING_PIGEON_VAR_NODE_H
