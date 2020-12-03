//
// Created by Theophile Champion on 28/11/2020.
//

#ifndef HOMING_PIGEON_ALGO_TREE_H
#define HOMING_PIGEON_ALGO_TREE_H

#include <memory>
#include <random>
#include <torch/torch.h>
#include <map>
#include "AlgoTreeConfig.h"

namespace hopi::graphs {
    class FactorGraph;
}
namespace hopi::nodes {
    class VarNode;
}
namespace hopi::distributions {
    class Distribution;
}

namespace hopi::algorithms {

    //
    // Definition of aliases
    //
    using VarNodePair = std::pair<nodes::VarNode*, nodes::VarNode*>;
    using NodeSelectionFn = nodes::VarNode *(AlgoTree::*)();
    using EvaluationFn = double (*)(nodes::VarNode *, nodes::VarNode *);

    /**
     * This class implements the tree search algorithm.
     */
    class AlgoTree {
    public:
        //
        // Factories
        //

        /**
         * Create a tree search algorithm for an agent having "n_acts" actions. The algorithm will use the vector
         * "state_pref" for the target distribution over hidden states and "obs_pref" for the target distribution
         * over observations.
         * @param n_acts the number of actions
         * @param state_pref the prior preference over hidden states
         * @param obs_pref the prior preference over observations
         * @return the tree search algorithm
         */
        static std::unique_ptr<AlgoTree> create(int n_acts, const torch::Tensor &&state_pref, const torch::Tensor &&obs_pref);

        /**
         * Create a tree search algorithm for an agent having "n_acts" actions. The algorithm will use the vector
         * "state_pref" for the target distribution over hidden states and "obs_pref" for the target distribution
         * over observations.
         * @param n_acts the number of actions
         * @param state_pref the prior preference over hidden states
         * @param obs_pref the prior preference over observations
         * @return the tree search algorithm
         */
        static std::unique_ptr<AlgoTree> create(int n_acts, const torch::Tensor &state_pref,  const torch::Tensor &obs_pref);

        /**
         * Create the tree search algorithm whose behaviour is described by the configuration "config".
         * @param config the configuration of the algorithm
         * @return the tree search algorithm
         */
        static std::unique_ptr<AlgoTree> create(AlgoTreeConfig &config);

    public:
        //
        // Constructors
        //

        /**
         * Construct a tree search algorithm for an agent having "n_acts" actions. The algorithm will use the vector
         * "state_pref" for the target distribution over hidden states and "obs_pref" for the target distribution
         * over observations.
         * @param n_acts the number of actions
         * @param state_pref the prior preference over hidden states
         * @param obs_pref the prior preference over observations
         */
        explicit AlgoTree(int n_acts, const torch::Tensor &&state_pref, const torch::Tensor &&obs_pref);

        /**
         * Construct a tree search algorithm for an agent having "n_acts" actions. The algorithm will use the vector
         * "state_pref" for the target distribution over hidden states and "obs_pref" for the target distribution
         * over observations.
         * @param n_acts the number of actions
         * @param state_pref the prior preference over hidden states
         * @param obs_pref the prior preference over observations
         */
        explicit AlgoTree(int n_acts, const torch::Tensor &state_pref,  const torch::Tensor &obs_pref);

        /**
         * Construct the tree search algorithm whose behaviour is described by the configuration "config".
         * @param config the configuration of the algorithm
         */
        explicit AlgoTree(AlgoTreeConfig &config);

        /**
         * Construct the tree search algorithm whose behaviour is described by the configuration "config".
         * @param config the configuration of the algorithm
         */
        explicit AlgoTree(AlgoTreeConfig &&config);

    public:
        //
        // Core functions
        //

        /**
         * Select the node to be expanded.
         * @param fg the factor graph on which the algo should be run
         * @return the node to be expanded
         */
        nodes::VarNode *nodeSelection(const std::shared_ptr<graphs::FactorGraph>& fg);

        /**
         * Perform an expansion of tree from node "n" using the matrices "A" and "B" as prior parameters for the
         * distributions P(O|S) and P(S|S_prev,U), where S_prev is the parent node of S, i.e., the previous state
         * of the system.
         * @param n the node to be expanded
         * @param A the likelihood matrix to use for the expansion
         * @param B the transition matrix to use for the expansion
         */
        void expansion(nodes::VarNode *n, const torch::Tensor &A, const torch::Tensor &B);

        /**
         * Perform an expansion of tree from node "n" using "A" and "B" that are assumed to be distributed according
         * to Dirichlet distributions.
         * @param n the node to be expanded
         * @param A the random matrix defining the likelihood mapping to use for the expansion
         * @param B the 3-tensor defining the transition mapping to use for the expansion
         */
        void expansion(nodes::VarNode *n, nodes::VarNode *A, nodes::VarNode *B);

        /**
         * Getter returning the state-observation pair corresponding to the last expansion.
         * @return the last state-observation pair that have been expanded
         */
        [[nodiscard]] std::vector<nodes::VarNode*> lastExpandedNodes() const;

        /**
         * Perform the evaluation of the last expanded nodes.
         */
        void evaluation();

        /**
         * Propagate the cost through the tree.
         * @param node the last hidden state that have been expanded
         * @param root the root of the tree
         */
        void propagation(nodes::VarNode *node, nodes::VarNode *root) const;

        /**
         * Select the action to be executed in the environment.
         * @param root the root of the tree
         * @return the action to be executed
         */
        static int actionSelection(nodes::VarNode *root);

    private:
        std::default_random_engine gen;    // Random number generator
        std::vector<VarNodePair> us;       // Unexplored states: state-observation pairs
        VarNodePair last_expansion;        // State-observation pair
        nodes::VarNode *tree_root;         // The node corresponding to S_t
        AlgoTreeConfig config;             // The configuration of the tree algorithm
        NodeSelectionFn nodeSelectionImpl; // Pointer on function implementing the node selection strategy
        EvaluationFn evaluationImpl;       // Pointer on function implementing the node evaluation strategy

    private:
        //
        // Auxiliary functions
        //

        /**
         * Getter returning the list of all actions that have not been expanded from n "n".
         * @param n the node for which the list of unexplored actions must be computed
         * @return the list of unexplored actions
         */
        std::vector<int> unexploredActions(nodes::VarNode *n) const;

        /**
         * Compare the cost of two state-observation pairs.
         * @param a1 the first state-observation pair
         * @param a2 the second state-observation pair
         * @return true if cost of "a1" is inferior to the cost of "a2" and false otherwise
         */
        static bool CompareCost(VarNodePair a1, VarNodePair a2);

        /**
         * Compute the distance between the node "n" and the root of the tree.
         * @param n the node from which the distance must be computed
         * @return the distance
         */
        int distanceFromRoot(nodes::VarNode *n);

    private:
        //
        // Configuration functions
        //

        /**
         * A getter returning a pointer on the function implementing the node selection strategy corresponding to
         * the type of node selection to use during the tree search.
         * @param type the type of node selection to use
         * @return the pointer on function implementing the node selection strategy
         */
        static NodeSelectionFn nodeSelectionFn(NodeSelectionType type);

        /**
         * A getter returning a pointer on the function implementing the node evaluation strategy corresponding to
         * the type of node selection to use during the tree search.
         * @param type the type of evaluation to use
         * @return the pointer on function implementing the node evaluation strategy
         */
        static EvaluationFn evaluationFn(EvaluationType type);

        //
        // Different kinds of evaluation
        //

        /**
         * Compute the quality of the state-observation pair (s,o) as: KL[Q(S)||V(S)] + KL[Q(O)||V(O)].
         * @param s the state to evaluate
         * @param o the observation to evaluate
         * @return the quality of the pair (s,o)
         */
        static double doubleKL(nodes::VarNode *s, nodes::VarNode *o);

        /**
         * Compute the quality of the state-observation pair (s,o) as: KL[Q(O)||V(O)] + E[P(O|S)], where the
         * expectation is with respect to P(O|S)Q(S). Note that this is the expected free energy of the pair (s,o).
         * @param s the state to evaluate
         * @param o the observation to evaluate
         * @return the quality of the pair (s,o)
         */
        static double efe(nodes::VarNode *s, nodes::VarNode *o);

        //
        // Different kinds of node selection
        //

        /**
         * Select the node with the lowest cost.
         * @return the node to be expanded
         */
        nodes::VarNode *nodeSelectionMin();

        /**
         * Select the node to be expanded by sampling a discrete distribution whose weights correspond to the quality
         * of the states that can still be expanded.
         * @return the node to be expanded
         */
        nodes::VarNode *nodeSelectionSampling();

        /**
         * Select the node to be expanded by sampling a softmax distribution whose weights correspond to the quality
         * of the states that can still be expanded.
         * @return the node to be expanded
         */
        nodes::VarNode *nodeSelectionSoftmaxSampling();
    };
}

#endif //HOMING_PIGEON_ALGO_TREE_H
