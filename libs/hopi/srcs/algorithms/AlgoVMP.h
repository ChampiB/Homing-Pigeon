//
// Created by Theophile Champion on 28/11/2020.
//

#ifndef HOMING_PIGEON_ALGO_VMP_H
#define HOMING_PIGEON_ALGO_VMP_H

#include <memory>
#include <vector>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::algorithms {

    /**
     * This class implement the Variational Message Passing algorithm used to perform inference of the latent variables.
     */
    class AlgoVMP {
    public:
        /**
         * Iterates the updates corresponding to the inputs variables. The iteration of the updates stops if:
         *  - the maximum number of iteration is reached;
         *  - or the Variational Free Energy has converged.
         * @param vars the input variables
         * @param epsilon the convergence threshold under which the VFE has converged
         * @param max_iter the maximum number of iterations
         */
        static void inference(const std::vector<nodes::VarNode*>& vars, double epsilon = 0.01, int max_iter = 2147483647);

        /**
         * Perform one iteration of the inference updates.
         * @param var the list of variables whose updates must be iterated
         */
        static void inference(nodes::VarNode *var);

        /**
         * Compute the Variational Free Energy (VFE) of the random variables sent as parameters.
         * @param vars the inputs variable on which the VFE must be computed
         * @return the VFE
         */
        static double vfe(const std::vector<nodes::VarNode*>& vars);
    };

}

#endif //HOMING_PIGEON_ALGO_VMP_H
