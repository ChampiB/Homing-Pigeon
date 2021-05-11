//
// Created by tmac3 on 19/03/2020.
//

#ifndef AIF_ADJACENT_FACTORS_ITER_H
#define AIF_ADJACENT_FACTORS_ITER_H

#include <vector>
#include <memory>

namespace hopi::nodes {
    class VarNode;
    class FactorNode;
}

namespace hopi::iterators {

    class AdjacentFactorsIter : public std::iterator<std::forward_iterator_tag,nodes::FactorNode*> {
    public:
        explicit AdjacentFactorsIter(nodes::VarNode *var);

    public:
        nodes::FactorNode *operator*();
        AdjacentFactorsIter &operator++();
        AdjacentFactorsIter &operator=(const AdjacentFactorsIter &other);
        bool operator==(const AdjacentFactorsIter &other) const;
        bool operator!=(const AdjacentFactorsIter &other) const;

    private:
        nodes::VarNode *_var;
        int _currentIndex;
    };

}

#endif //AIF_ADJACENT_FACTORS_ITER_H
