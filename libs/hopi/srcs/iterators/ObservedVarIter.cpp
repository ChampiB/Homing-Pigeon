//
// Created by Theophile Champion on 04/12/2020.
//

#include "ObservedVarIter.h"
#include "graphs/FactorGraph.h"
#include "nodes/VarNode.h"
#include "nodes/VarNodeType.h"

using namespace hopi::nodes;

namespace hopi::iterators {

    ObservedVarIter::ObservedVarIter(hopi::graphs::FactorGraph *fg) {
        _currentIndex = 0;
        _currVarIndex = 0;
        _fg = fg;
        nextObservedVar();
    }

    hopi::nodes::VarNode *ObservedVarIter::operator*() {
        if (_currentIndex < _fg->nObservedVar()) {
            return _fg->node(_currVarIndex);
        } else {
            return nullptr;
        }
    }

    ObservedVarIter &ObservedVarIter::operator++() {
        ++_currentIndex;
        ++_currVarIndex;
        nextObservedVar();
        return *this;
    }

    ObservedVarIter &ObservedVarIter::operator=(const ObservedVarIter &other) {
        if(this == &other)
            return *this;
        _fg = other._fg;
        _currentIndex = other._currentIndex;
        _currVarIndex = other._currVarIndex;
        return *this;
    }

    bool ObservedVarIter::operator==(const ObservedVarIter &other) const {
        return _currentIndex == other._currentIndex;
    }

    bool ObservedVarIter::operator!=(const ObservedVarIter &other) const {
        return !(*this == other);
    }

    void ObservedVarIter::nextObservedVar() {
        if (_currentIndex < _fg->nObservedVar()) {
            while (_currVarIndex < _fg->nodes() && _fg->node(_currVarIndex)->type() != VarNodeType::OBSERVED) {
                ++_currVarIndex;
            }
        }
    }

}
