//
// Created by Theophile Champion on 19/03/2020.
//

#ifndef HOMING_PIGEON_ADJACENT_FACTORS_ITER_H
#define HOMING_PIGEON_ADJACENT_FACTORS_ITER_H

#include <vector>
#include <memory>

namespace hopi::nodes {
    class VarNode;
    class FactorNode;
}

namespace hopi::iterators {

    /**
     * An iterator over the adjacent factors of a random variable.
     */
    class AdjacentFactorsIter : public std::iterator<std::forward_iterator_tag,nodes::FactorNode*> {
    public:
        //
        // Constructor
        //

        /**
         * Construct an iterator over the adjacent factors of a random variable.
         * @param var the variable whose adjacent factors will be iterated
         */
        explicit AdjacentFactorsIter(nodes::VarNode *var);

    public:
        /**
         * Getter.
         * @return the current factor.
         */
        nodes::FactorNode *operator*();

        /**
         * Go to the next adjacent factor.
         * @return this
         */
        AdjacentFactorsIter &operator++();

        /**
         * Copy constructor.
         * @param other another iterator over adjacent factors
         * @return this
         */
        AdjacentFactorsIter &operator=(const AdjacentFactorsIter &other);

        /**
         * Check whether this and other are currently pointing to same adjacent factor.
         * @param other another iterator
         * @return true if the two iterators point to the same factor and false otherwise
         */
        bool operator==(const AdjacentFactorsIter &other) const;

        /**
         * Check whether this and other are currently pointing to different adjacent factor.
         * @param other another iterator
         * @return false if the two iterators point to the same factor and true otherwise
         */
        bool operator!=(const AdjacentFactorsIter &other) const;

    private:
        nodes::VarNode *_var;
        int _currentIndex;
    };

}

#endif //HOMING_PIGEON_ADJACENT_FACTORS_ITER_H
