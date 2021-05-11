//
// Created by tmac3 on 19/03/2020.
//

#ifndef AIF_HIDDEN_VAR_ITER_H
#define AIF_HIDDEN_VAR_ITER_H

#include <vector>
#include <memory>

namespace hopi::nodes {
    class VarNode;
}
namespace hopi::graphs{
    class FactorGraph;
}

namespace hopi::iterators {

    class HiddenVarIter : public std::iterator<std::forward_iterator_tag,nodes::VarNode*> {
    public:
        explicit HiddenVarIter(const std::vector<nodes::VarNode*>& vars);

    public:
        nodes::VarNode *operator*();
        HiddenVarIter &operator++();
        HiddenVarIter &operator=(const HiddenVarIter &other);
        bool operator==(const HiddenVarIter &other) const;
        bool operator!=(const HiddenVarIter &other) const;

    private:
        void nextHiddenVar();

    private:
        std::vector<nodes::VarNode*> _vars;
        int _currentIndex;
        int _currVarIndex;
        int _nHiddenStates;
    };

}

#endif //AIF_HIDDEN_VAR_ITER_H
