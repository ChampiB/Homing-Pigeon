//
// Created by Theophile Champion on 03/08/2021.
//

#ifndef EXPERIMENTS_AI_TS_MCTS_H
#define EXPERIMENTS_AI_TS_MCTS_H

#include <memory>
#include <torch/torch.h>
#include "EvaluationType.h"

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::algorithms::planning {

    class MCTSConfig;

    typedef double (*EvaluationFunction)(
            const torch::Tensor &sBeliefs,
            const torch::Tensor &oBeliefs,
            const torch::Tensor &a,
            const std::shared_ptr<MCTSConfig> &c
            );

    /**
     * A class implementing the MCTS algorithm.
     */
    class MCTS {
    public:
        /**
         * Create a Monte Carlo Tree Search algorithm.
         * @param config the configuration of the MCTS algorithm.
         * @return the MCTS algorithm.
         */
        static std::unique_ptr<MCTS> create(const std::shared_ptr<MCTSConfig> &config);

        /**
         * Constructor.
         * @param config the configuration of the MCTS algorithm.
         */
        explicit MCTS(const std::shared_ptr<MCTSConfig> &config);

        /**
         * Select the node to be expanded.
         * @param root of the tree on which MCTS is run.
         * @param nbActions the number of actions in the environment.
         * @return the selected node.
         */
        [[nodiscard]] hopi::nodes::VarNode *selectNode(hopi::nodes::VarNode *root, int nbActions) const;

        /**
         * Perform an expansion of the selected node.
         * @param node the node selected for expansion.
         * @param a the likelihood mapping.
         * @param b the transition mapping.
         * @return the list of newly expanded nodes.
         */
        static std::vector<hopi::nodes::VarNode*> expansion(hopi::nodes::VarNode *node, const torch::Tensor &a, const torch::Tensor &b);

        /**
         * Evaluate the cost of all expanded nodes.
         * @param nodes the newly expanded nodes.
         * @param a the likelihood mapping.
         * @param type the evaluation function to be used.
         */
        void evaluation(
                const std::vector<hopi::nodes::VarNode*> &nodes,
                const torch::Tensor &a,
                const EvaluationType &type
        );

        /**
         * Propagate the cost of the newly expanded nodes and update the number of visits.
         * @param nodes the newly expanded nodes.
         */
        static void propagation(const std::vector<hopi::nodes::VarNode*> &nodes) ;

        /**
         * Select the action to be performed.
         * @param root of the tree on which MCTS is run.
         * @return the selected action.
         */
        [[nodiscard]] int selectAction(hopi::nodes::VarNode *root) const;

        /**
         * Getter.
         * @return the configuration of the MCTS algorithm.
         */
        [[nodiscard]] std::shared_ptr<MCTSConfig> config() const;

    private:
        /**
         * Compare the cost of the two input nodes.
         * @param n1 first input node.
         * @param n2 second input node.
         * @return true if n1 is less than n2, false otherwise.
         */
        static bool compareCost(hopi::nodes::VarNode *n1, hopi::nodes::VarNode *n2);

        /**
         * Compare the uct criterion of the two input nodes.
         * @param n1 first input node.
         * @param n2 second input node.
         * @return true if n1 is less than n2, false otherwise.
         */
        [[nodiscard]] bool compareUCT(hopi::nodes::VarNode *n1, hopi::nodes::VarNode *n2) const;

        /**
         * Compute the uct criterion of the input node.
         * @param node the node whose uct criterion must be computed.
         * @return the uct criterion.
         */
        [[nodiscard]] double uct(hopi::nodes::VarNode *node) const;

        /**
         * Compute the expected free energy of a node.
         * @param sBeliefs posterior beliefs over states.
         * @param oBeliefs posterior beliefs over observations.
         * @param a the likelihood mapping.
         * @param c the configuration of the MCTS algorithm.
         * @return the expected free energy.
         */
        static double efe(
                const torch::Tensor &sBeliefs,
                const torch::Tensor &oBeliefs,
                const torch::Tensor &a,
                const std::shared_ptr<MCTSConfig> &c
        );

        /**
         * Compute the pure cost of a node, i.e. the sum of the risk over observations and risk over states.
         * @param sBeliefs posterior beliefs over states.
         * @param oBeliefs posterior beliefs over observations.
         * @param a the likelihood mapping.
         * @param c the configuration of the MCTS algorithm.
         * @return the expected free energy.
         */
        static double doubleKL(
                const torch::Tensor &sBeliefs,
                const torch::Tensor &oBeliefs,
                const torch::Tensor &a,
                const std::shared_ptr<MCTSConfig> &c
        );

        /**
         * Getter.
         * @param node whose parent should be returned.
         * @return the parent of the input node.
         */
        static hopi::nodes::VarNode *parent(hopi::nodes::VarNode *node);

    private:
        std::shared_ptr<MCTSConfig> _config;
    };

}

#endif //EXPERIMENTS_AI_TS_MCTS_H
