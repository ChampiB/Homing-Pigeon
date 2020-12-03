//
// Created by Theophile Champion on 19/03/2020.
//

#include "HiddenVarIter.h"
#include "graphs/FactorGraph.h"
#include "nodes/VarNode.h"

using namespace hopi::graphs;
using namespace hopi::nodes;

namespace hopi::iterators {

    HiddenVarIter::HiddenVarIter(const std::vector<nodes::VarNode*>& vars) {
        _currentIndex = 0;
        _currVarIndex = 0;
        _vars = vars;
        _nHiddenStates = (int) std::count_if(_vars.begin(), _vars.end(), [](const VarNode *elem) {
            return (elem->type() == VarNodeType::HIDDEN);
        });
        nextHiddenVar();
    }

    void HiddenVarIter::nextHiddenVar() {
        if (_currentIndex < _nHiddenStates) {
            while (_currVarIndex < _vars.size() && _vars[_currVarIndex]->type() != VarNodeType::HIDDEN) {
                ++_currVarIndex;
            }
        }
    }

    VarNode *HiddenVarIter::operator*() {
        if (_currentIndex < _nHiddenStates) {
            return _vars[_currVarIndex];
        } else {
            return nullptr;
        }
    }

    HiddenVarIter &HiddenVarIter::operator++() {
        ++_currentIndex;
        ++_currVarIndex;
        nextHiddenVar();
        return *this;
    }

    HiddenVarIter &HiddenVarIter::operator=(const HiddenVarIter &other) {
        if(this == &other)
            return *this;
        _vars = other._vars;
        _currentIndex = other._currentIndex;
        _currVarIndex = other._currVarIndex;
        return *this;
    }

    bool HiddenVarIter::operator==(const HiddenVarIter &other) const {
        return _currentIndex == other._currentIndex;
    }

    bool HiddenVarIter::operator!=(const HiddenVarIter &other) const {
        return !(*this == other);
    }

}
