//
// Created by Theophile Champion on 28/11/2020.
//

#include "VarNode.h"

#include <utility>
#include "distributions/Distribution.h"

using namespace hopi::distributions;

namespace hopi::nodes {

    std::unique_ptr<VarNode> VarNode::create(VarNodeType type) {
        return std::make_unique<VarNode>(type);
    }

    VarNode::VarNode(VarNodeType type) : N(0), G(-1), _type(type), A(-1),
        _parent(nullptr), _prior(nullptr), _posterior(nullptr), _biased(nullptr) {}

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

    void VarNode::setAction(int a) {
        A = a;
    }

    void VarNode::incrementN() {
        N += 1;
    }

    FactorNode *VarNode::parent() {
        return _parent;
    }

    std::vector<FactorNode *>::iterator VarNode::firstChild() {
        return _children.begin();
    }

    std::vector<FactorNode *>::iterator VarNode::lastChild() {
        return _children.end();
    }

    int VarNode::action() const {
        return A;
    }

    int VarNode::n() const {
        return N;
    }

    double VarNode::g() const {
        return G;
    }

    void VarNode::setG(double g) {
        G = g;
    }

    Distribution *VarNode::posterior() {
        return _posterior.get();
    }

    Distribution *VarNode::biased() {
        return _biased.get();
    }

    int VarNode::nChildren() const {
        return (int) _children.size();
    }

    VarNodeType VarNode::type() const {
        return _type;
    }

    distributions::Distribution *VarNode::prior() {
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

}
