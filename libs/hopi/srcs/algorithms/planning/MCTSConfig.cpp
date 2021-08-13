//
// Created by Theophile Champion on 03/08/2021.
//

#include <iostream>
#include "MCTSConfig.h"

using namespace torch;

namespace hopi::algorithms::planning {

    std::shared_ptr<MCTSConfig> MCTSConfig::create(
            const Tensor &obsPref,
            const Tensor &statePref,
            int planningSteps,
            double expConst,
            double prefPrecision,
            double actionPrecision
    ) {
        return std::make_shared<MCTSConfig>(
                obsPref, statePref, planningSteps, expConst, prefPrecision, actionPrecision
        );
    }

    MCTSConfig::MCTSConfig(
            const Tensor &obsPref,
            const Tensor &statePref,
            int planningSteps,
            double expConst,
            double prefPrecision,
            double actionPrecision
    ) {
        _planningSteps = planningSteps;
        _expConst = expConst;
        _cPrecision = prefPrecision;
        _obsPref = softmax(obsPref * prefPrecision, 0);
        _statePref = softmax(statePref * prefPrecision, 0);
        _aPrecision = actionPrecision;
    }

    double MCTSConfig::explorationConstant() const {
        return _expConst;
    }

    double MCTSConfig::actionPrecision() const {
        return _aPrecision;
    }

    int MCTSConfig::nbPlanningSteps() const {
        return _planningSteps;
    }

    void MCTSConfig::setPlanningSteps(int value) {
        _planningSteps = value;
    }

    void MCTSConfig::setPrecisionPreferences(double value) {
        _cPrecision = value;
    }

    void MCTSConfig::setPrecisionActionSelection(double value) {
        _aPrecision = value;
    }

    void MCTSConfig::setExplorationConstant(double value) {
        _expConst = value;
    }

    void MCTSConfig::setStatesPreferences(const Tensor &statePref) {
        _statePref = statePref;
    }

    torch::Tensor MCTSConfig::obsPreferences() const {
        return _obsPref;
    }

    torch::Tensor MCTSConfig::statesPreferences() const {
        return _statePref;
    }

    void MCTSConfig::print(std::ostream &output) const {
        output << "========== MCTS CONFIGURATION ==========" << std::endl;
        output << "Exploration constant: " << _expConst << std::endl;
        output << "Precision over actions: " << _aPrecision << std::endl;
        output << "Precision over prior preferences: " << _cPrecision << std::endl;
        output << "Prior preferences over observations: " << _obsPref << std::endl;
        output << "Prior preferences over hidden states: " << _statePref << std::endl;
        output << "Number of planning iterations: " << _planningSteps << std::endl;
        output << std::endl;
    }

}
