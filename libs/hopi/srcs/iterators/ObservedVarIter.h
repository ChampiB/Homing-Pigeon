//
// Created by Theophile Champion on 04/12/2020.
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

    /**
     * An iterator over the observed variable of a factor graph.
     */
    class ObservedVarIter : public std::iterator<std::forward_iterator_tag,nodes::VarNode*> {
    public:
        //
        // Constructor
        //

        /**
         * Construct an iterator over observed variable
         * @param fg the factor graph on which the iterator will operate
         */
        explicit ObservedVarIter(graphs::FactorGraph *fg);

    public:
        /**
         * Getter.
         * @return the current observed variable.
         */
        nodes::VarNode *operator*();

        /**
         * Go to the next observed variable in the factor graph.
         * @return this
         */
        ObservedVarIter &operator++();

        /**
         * Copy constructor.
         * @param other another iterator of observed variable
         * @return this
         */
        ObservedVarIter &operator=(const ObservedVarIter &other);

        /**
         * Check whether this and other are currently pointing to same observed variable.
         * @param other another iterator
         * @return true if the two iterators point to the same variable and false otherwise
         */
        bool operator==(const ObservedVarIter &other) const;

        /**
         * Check whether this and other are currently pointing to different observed variable.
         * @param other another iterator
         * @return false if the two iterators point to the same variable and true otherwise
         */
        bool operator!=(const ObservedVarIter &other) const;

    private:
        /**
         * Go to the next observed variable in the factor graph.
         */
        void nextObservedVar();

    private:
        graphs::FactorGraph *_fg;
        int _currentIndex;
        int _currVarIndex;
    };

}

#endif //HOMING_PIGEON_2_OBSERVEDVARITER_H
