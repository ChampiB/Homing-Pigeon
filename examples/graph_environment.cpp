#include "environments/GraphEnv.h"

using namespace hopi::environments;

int main() {
    /**
     ** Create the environment.
     **/
    auto env = GraphEnv::create(2, 3, {4, 2});

    /**
     ** Run some simulation showing how the environment behave.
     **/
    std::cout << "Simulation 1: the longest path does not lead to the bad state." << std::endl;
    env->print();
    for (int i = 0; i < 7; ++i) {
        env->execute(0);
        env->print();
    }
    env->reset();

    std::cout << "Simulation 2: the shortest path lead to the bad state." << std::endl;
    env->print();
    env->execute(1);
    env->print();
    for (int i = 0; i < 4; ++i) {
        env->execute(0);
        env->print();
    }
    env->reset();

    std::cout << "Simulation 3: bad actions from initial state lead to the bad state." << std::endl;
    env->print();
    env->execute(2);
    env->print();
    env->reset();

    std::cout << "Simulation 4: bad actions within a path lead to the bad state." << std::endl;
    env->print();
    env->execute(0);
    env->print();
    env->execute(2);
    env->print();
    env->reset();

    return EXIT_SUCCESS;
}
