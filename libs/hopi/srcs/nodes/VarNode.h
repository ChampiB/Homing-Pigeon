//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_VARNODE_H
#define HOMING_PIGEON_2_VARNODE_H

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

namespace hopi::nodes {

    class VarNode {
    public:
        static std::unique_ptr<VarNode> create(VarNodeType type);

    public:
        explicit VarNode(VarNodeType type);
        void setPrior(std::unique_ptr<distributions::Distribution> prior);
        void setPosterior(std::unique_ptr<distributions::Distribution> posterior);
        void setBiased(std::unique_ptr<distributions::Distribution> biased);
        void setParent(FactorNode *parent);
        void setAction(int action);
        void setG(double g);
        void setType(VarNodeType type);
        void setName(std::string &name);
        void setName(std::string &&name);
        void incrementN();
        FactorNode *parent();
        std::vector<FactorNode *>::iterator firstChild();
        std::vector<FactorNode *>::iterator lastChild();
        void addChild(FactorNode *child);
        [[nodiscard]] int nChildren() const;
        [[nodiscard]] int action() const;
        [[nodiscard]] double g() const;
        [[nodiscard]] int n() const;
        [[nodiscard]] VarNodeType type() const;
        [[nodiscard]] std::string name() const;
        distributions::Distribution *prior();
        distributions::Distribution *posterior();
        distributions::Distribution *biased();
        void removeNullChildren();
        void disconnectChild(nodes::FactorNode *node);

    private:
        int N;    // Number of children expanded   (only used within the tree)
        double G; // Node's cost / inverse quality (only used within the tree)
        int A;    // Action that led to this state (only used within the tree)
        std::vector<FactorNode *> _children;
        FactorNode *_parent;
        std::string _name;
        std::unique_ptr<distributions::Distribution> _prior;
        std::unique_ptr<distributions::Distribution> _posterior; // Evidence for OBSERVED variables
        std::unique_ptr<distributions::Distribution> _biased;    // Prior preferences
        VarNodeType _type;
    };

}

#endif //HOMING_PIGEON_2_VARNODE_H
