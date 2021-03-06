//
// Created by Theophile Champion on 28/11/2020.
//

#include "nodes/FactorNode.h"

namespace hopi::nodes {

    std::string FactorNode::name() const {
        return _name;
    }

    void FactorNode::setName(std::string &name) {
        _name = name;
    }

    void FactorNode::setName(std::string &&name) {
        _name = name;
    }

}
