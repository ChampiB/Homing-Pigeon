//
// Created by tmac3 on 28/11/2020.
//

#include "nodes/FactorNode.h"

#include <utility>

namespace hopi::nodes {

    std::string FactorNode::name() const {
        return _name;
    }

    void FactorNode::setName(std::string name) {
        _name = std::move(name);
    }

}
