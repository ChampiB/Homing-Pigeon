//
// Created by Theophile Champion on 28/11/2020.
//

#include "VarNode.h"
#include "FactorNode.h"
#include <utility>
#include "distributions/Distribution.h"
#include "algorithms/planning/MCTSNodeData.h"

using namespace hopi::distributions;
using namespace hopi::algorithms::planning;

namespace hopi::nodes {

    std::unique_ptr<VarNode> VarNode::create(VarNodeType type) {
        return std::make_unique<VarNode>(type);
    }

    VarNode::VarNode(VarNodeType type) :
        _type(type), _parent(nullptr), _prior(nullptr), _posterior(nullptr), _biased(nullptr) {
        _data = MCTSNodeData::create();
    }

    VarNode::~VarNode() = default;

    void VarNode::setPrior(std::unique_ptr<Distribution> p) {
        _prior = std::move(p);
    }

    void VarNode::setPosterior(std::unique_ptr<Distribution> p) {
        _posterior = std::move(p);
    }

    void VarNode::setBiased(std::unique_ptr<Distribution> b) {
        _biased = std::move(b);
    }

    void VarNode::setParent(FactorNode *p) {
        _parent = p;
    }

    FactorNode *VarNode::parent() const {
        return _parent;
    }

    std::vector<FactorNode *>::iterator VarNode::firstChild() {
        return _children.begin();
    }

    std::vector<FactorNode *>::iterator VarNode::lastChild() {
        return _children.end();
    }

    Distribution *VarNode::posterior() const {
        return _posterior.get();
    }

    Distribution *VarNode::biased() const {
        return _biased.get();
    }

    int VarNode::nChildren() const {
        return (int) _children.size();
    }

    VarNodeType VarNode::type() const {
        return _type;
    }

    distributions::Distribution *VarNode::prior() const {
        return _prior.get();
    }

    void VarNode::setType(VarNodeType type) {
        _type = type;
    }

    void VarNode::addChild(FactorNode *c) {
        _children.push_back(c);
    }

    void VarNode::setName(std::string &name) {
        _name = name;
    }

    void VarNode::setName(std::string &&name) {
        _name = name;
    }

    std::string VarNode::name() const {
        return _name;
    }

    MCTSNodeData *VarNode::data() const {
        return _data.get();
    }

    void VarNode::removeNullChildren() {
        _children.erase(std::remove_if(_children.begin(), _children.end(),
                                       [](FactorNode * &x){return x == nullptr;}), _children.end());
    }

    void VarNode::disconnectChild(nodes::FactorNode *node) {
        std::vector<FactorNode*>::iterator it;

        it = std::find(_children.begin(), _children.end(), node);
        if (it != _children.end()) {
            *it = nullptr;
        }
    }

    VarNode *VarNode::child(int id) const {
        return _children[id]->child();
    }

    int VarNode::nChildrenHiddenStates() const {
        return (int) std::count_if(_children.begin(), _children.end(), [](FactorNode *node) {
            return node->child()->data()->action != -1;
        });
    }

}
