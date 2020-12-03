//
// Created by Theophile Champion on 19/03/2020.
//

#include "AdjacentFactorsIter.h"
#include "nodes/VarNode.h"

using namespace hopi::nodes;

namespace hopi::iterators {

    AdjacentFactorsIter::AdjacentFactorsIter(VarNode *var) {
        _currentIndex = 0;
        _var = var;
    }

    FactorNode *AdjacentFactorsIter::operator*() {
        if (_currentIndex < _var->nChildren()) {
            return *(_var->firstChild() + _currentIndex);
        } else if (_currentIndex == _var->nChildren()) {
            return _var->parent();
        } else {
            return nullptr;
        }
    }

    AdjacentFactorsIter &AdjacentFactorsIter::operator++() {
        ++_currentIndex;
        return *this;
    }

    AdjacentFactorsIter &AdjacentFactorsIter::operator=(const AdjacentFactorsIter &other) {
        if(this == &other)
            return *this;
        _var = other._var;
        _currentIndex = other._currentIndex;
        return *this;
    }

    bool AdjacentFactorsIter::operator==(const AdjacentFactorsIter &other) const {
        return _currentIndex == other._currentIndex;
    }

    bool AdjacentFactorsIter::operator!=(const AdjacentFactorsIter &other) const {
        return !(*this == other);
    }

}
