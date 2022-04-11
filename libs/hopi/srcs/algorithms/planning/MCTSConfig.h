//
// Created by Theophile Champion on 03/08/2021.
//

#ifndef EXPERIMENTS_AI_TS_MCTS_V2_CONFIG_H
#define EXPERIMENTS_AI_TS_MCTS_V2_CONFIG_H

#include <torch/torch.h>
#include <memory>

namespace hopi::algorithms::planning {

    /**
     * A class storing the configuration of the MCTS algorithm.
     */
    class MCTSConfig {
    public:
        /**
         * Create a configuration for the MCTS algorithm.
         * @param obsPref the preferences over observations.
         * @param statePref the preferences over states.
         * @param planningSteps number of planning iteration.
         * @param expConst exploration constant.
         * @param prefPrecision precision over prior preferences.
         * @param actionPrecision precision over actions.
         * @return the configuration.
         */
        static std::shared_ptr<MCTSConfig> create(
                const torch::Tensor &obsPref,
                const torch::Tensor &statePref,
                int planningSteps,
                double expConst,
                double prefPrecision,
                double actionPrecision
        );

        /**
         * Constructor.
         * @param obsPref the preferences over observations.
         * @param statePref the preferences over states.
         * @param planningSteps number of planning iteration.
         * @param expConst exploration constant.
         * @param prefPrecision precision over prior preferences.
         * @param actionPrecision precision over actions.
         */
        MCTSConfig(
                const torch::Tensor &obsPref,
                const torch::Tensor &statePref,
                int planningSteps,
                double expConst,
                double prefPrecision,
                double actionPrecision
        );

        /**
         * Getter.
         * @return the exploration constant of the MCTS algorithm.
         */
        [[nodiscard]] double explorationConstant() const;

        /**
         * Getter.
         * @return the precision over actions of the MCTS algorithm.
         */
        [[nodiscard]] double actionPrecision() const;

        /**
         * Getter.
         * @return the preferences over observations.
         */
        [[nodiscard]] torch::Tensor obsPreferences() const;

        /**
         * Getter.
         * @return the preferences over states.
         */
        [[nodiscard]] torch::Tensor statesPreferences() const;

        /**
         * Getter.
         * @return the number of planning iterations.
         */
        [[nodiscard]] int nbPlanningSteps() const;

        /**
         * Setter.
         * @param statePref the new prior preferences over hidden states.
         */
        void setStatesPreferences(const torch::Tensor &statePref);

        /**
         * Setter.
         * @param value new number of planning iteration.
         */
        void setPlanningSteps(int value);

        /**
         * Setter.
         * @param value new precision over prior preferences.
         */
        void setPrecisionPreferences(double value);

        /**
         * Setter.
         * @param value new precision over action.
         */
        void setPrecisionActionSelection(double value);

        /**
         * Setter.
         * @param value new exploration constant.
         */
        void setExplorationConstant(double value);

        /**
         * Print the configuration in the output stream.
         * @param output the stream
         */
        void print(std::ostream &output) const;

    private:
        double _expConst;
        double _aPrecision;
        double _cPrecision;
        int _planningSteps;
        torch::Tensor _obsPref;
        torch::Tensor _statePref;
    };

}

#endif //EXPERIMENTS_AI_TS_MCTS_V2_CONFIG_H
