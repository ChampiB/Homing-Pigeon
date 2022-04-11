//
// Created by Theophile Champion on 16/07/2021.
//

#ifndef EXPERIMENTS_AI_TS_MCTS_NODE_DATA_H
#define EXPERIMENTS_AI_TS_MCTS_NODE_DATA_H

#include <memory>

namespace hopi::algorithms::planning {

    /**
     * A class storing the data of a node used by the MCTS algorithm.
     */
    class MCTSNodeData {
    public:
        /**
         * Create default node's data for MCTS planning algorithm;
         * @return the created node's data
         */
        static std::unique_ptr<MCTSNodeData> create();

        /**
         * Constructor of the node's data for MCTS.
         * @param N the number of visits
         * @param G the cost of the node
         * @param A the action that led to the node
         * @param pruned true if the node should be pruned, false otherwise
         */
        explicit MCTSNodeData(int N = 0, double G = -1, int A = -1, bool pruned = false);

        /**
         * Destruct the node's data.
         */
        ~MCTSNodeData() = default;

    public:
        int    visits;     // Number of visits
        double cost;       // Node's total cost
        int    action;     // Action that led to this state
        bool   pruned;     // Should this branch be discarded during node selection?
    };

}

#endif //EXPERIMENTS_AI_TS_MCTS_NODE_DATA_H
