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

    class AdjacentFactorsIter : public std::iterator<std::forward_iterator_tag,hopi::nodes::FactorNode*> {
    public:
        explicit AdjacentFactorsIter(hopi::nodes::VarNode *var);

    public:
        hopi::nodes::FactorNode *operator*();
        AdjacentFactorsIter &operator++();
        AdjacentFactorsIter &operator=(const AdjacentFactorsIter &other);
        bool operator==(const AdjacentFactorsIter &other) const;
        bool operator!=(const AdjacentFactorsIter &other) const;

    private:
        hopi::nodes::VarNode *_var;
        int _currentIndex;
    };

}

#endif //AIF_ADJACENT_FACTORS_ITER_H
