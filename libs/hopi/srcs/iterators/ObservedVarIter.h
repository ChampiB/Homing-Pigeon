//
// Created by tmac3 on 04/12/2020.
//

#ifndef HOMING_PIGEON_2_OBSERVEDVARITER_H
#define HOMING_PIGEON_2_OBSERVEDVARITER_H

#include <vector>
#include <memory>

namespace hopi::nodes {
    class VarNode;
}
namespace hopi::graphs{
    class FactorGraph;
}

namespace hopi::iterators {

    class ObservedVarIter : public std::iterator<std::forward_iterator_tag,hopi::nodes::VarNode*> {
    public:
        explicit ObservedVarIter(hopi::graphs::FactorGraph *fg);

    public:
        hopi::nodes::VarNode *operator*();
        ObservedVarIter &operator++();
        ObservedVarIter &operator=(const ObservedVarIter &other);
        bool operator==(const ObservedVarIter &other) const;
        bool operator!=(const ObservedVarIter &other) const;

    private:
        void nextObservedVar();

    private:
        hopi::graphs::FactorGraph *_fg;
        int _currentIndex;
        int _currVarIndex;
    };

}

#endif //HOMING_PIGEON_2_OBSERVEDVARITER_H
