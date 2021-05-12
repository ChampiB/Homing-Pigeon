//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include "algorithms/AlgoVMP.h"
#include "graphs/FactorGraph.h"
#include "math/Ops.h"
#include "contexts/FactorGraphContexts.h"
#include "nodes/VarNode.h"
#include "nodes/FactorNode.h"
#include "distributions/Categorical.h"
#include "helpers/UnitTests.h"
#include "api/API.h"
#include <iostream>

using namespace torch;
using namespace hopi::algorithms;
using namespace hopi::distributions;
using namespace hopi::graphs;
using namespace hopi::api;
using namespace hopi::nodes;
using namespace hopi::math;
using namespace tests;

TEST_CASE( "Inference process stop when vfe has converged (according to epsilon)" ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto vars = fg->getNodes();

        AlgoVMP::inference(vars, 0.1);
        double beforeF = AlgoVMP::vfe(vars);
        AlgoVMP::inference(vars[0]);
        double afterF = AlgoVMP::vfe(vars);
        REQUIRE( afterF - beforeF < 0.1 );
    });
}

TEST_CASE( "During inference the vfe always decreases" ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto vars = fg->getNodes();

        int k = 0;
        do {
            for (int i = 0; i < vars.size(); ++i) {
                if (vars[i]->type() == HIDDEN) {
                    double beforeF = AlgoVMP::vfe(vars);
                    AlgoVMP::inference(vars[i]);
                    double afterF = AlgoVMP::vfe(vars);
                    REQUIRE(beforeF >= afterF);
                }
            }
        } while (++k < 3);
    });
}

TEST_CASE( "AlgoVMP.vfe() returns the variational free energy of the variables sent as parameters" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto fg     = FactorGraph::current();
        VarNode *a0 = API::Categorical(Ops::uniformColumnWise({2}));
        VarNode *s0 = API::Categorical(Ops::uniformColumnWise({5}));
        VarNode *s1 = API::ActiveTransition(s0, a0, Ops::uniformColumnWise({2,5,5}));
        s1->setType(OBSERVED);

        double F = AlgoVMP::vfe(fg->getNodes());
        double res = 0;
        res += a0->parent()->vfe();
        res += s0->parent()->vfe();
        res += s1->parent()->vfe();
        REQUIRE( F == Approx(res).epsilon(0.1) );
    });
}
