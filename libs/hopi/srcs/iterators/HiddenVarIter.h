//
// Created by Theophile Champion on 19/03/2020.
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

    /**
     * An iterator over the hidden variable that works on a list of variables.
     */
    class HiddenVarIter : public std::iterator<std::forward_iterator_tag,nodes::VarNode*> {
    public:
        //
        // Constructor
        //

        /**
         * Construct an iterator over hidden variable
         * @param vars the list of variables on which the iterator will operate
         */
        explicit HiddenVarIter(const std::vector<nodes::VarNode*>& vars);

    public:
        /**
         * Getter.
         * @return the current hidden variable.
         */
        nodes::VarNode *operator*();

        /**
         * Go to the next hidden variable in the list of variables.
         * @return this
         */
        HiddenVarIter &operator++();

        /**
         * Copy constructor.
         * @param other another iterator of hidden variable
         * @return this
         */
        HiddenVarIter &operator=(const HiddenVarIter &other);

        /**
         * Check whether this and other are currently pointing to same hidden variable.
         * @param other another iterator
         * @return true if the two iterators point to the same variable and false otherwise
         */
        bool operator==(const HiddenVarIter &other) const;

        /**
         * Check whether this and other are currently pointing to different hidden variable.
         * @param other another iterator
         * @return false if the two iterators point to the same variable and true otherwise
         */
        bool operator!=(const HiddenVarIter &other) const;

    private:
        /**
         * Go to the next hidden variable in the list of variables.
         */
        void nextHiddenVar();

    private:
        std::vector<nodes::VarNode*> _vars;
        int _currentIndex;
        int _currVarIndex;
        int _nHiddenStates;
    };

}

#endif //AIF_HIDDEN_VAR_ITER_H
