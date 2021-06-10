//
// Created by Theophile Champion on 28/11/2020.
//

#include <fstream>
#include <distributions/Dirichlet.h>
#include "distributions/Categorical.h"
#include "nodes/VarNode.h"
#include "nodes/FactorNode.h"
#include "math/Ops.h"
#include "api/API.h"
#include "FactorGraph.h"
#include "iterators/ObservedVarIter.h"

using namespace hopi::nodes;
using namespace hopi::iterators;
using namespace hopi::distributions;
using namespace hopi::math;
using namespace hopi::api;
using namespace torch;
using namespace torch::detail;

namespace hopi::graphs {

    static std::shared_ptr<FactorGraph> currentFactorGraph = nullptr;

    std::shared_ptr<FactorGraph> FactorGraph::current() {
        if (currentFactorGraph == nullptr)
            currentFactorGraph = std::make_shared<FactorGraph>();
        return currentFactorGraph;
    }

    void FactorGraph::setCurrent(std::shared_ptr<FactorGraph> &ptr) {
        currentFactorGraph = ptr;
    }

    void FactorGraph::setCurrent(std::shared_ptr<FactorGraph> &&ptr) {
        currentFactorGraph = ptr;
    }

    FactorGraph::FactorGraph() : _tree_root(nullptr) {}

    VarNode *FactorGraph::addNode(std::unique_ptr<VarNode> node) {
        _vars.push_back(std::move(node));
        return _vars[_vars.size() - 1].get();
    }

    FactorNode *FactorGraph::addFactor(std::unique_ptr<FactorNode> factor) {
        _factors.push_back(std::move(factor));
        return _factors[_factors.size() - 1].get();
    }

    void FactorGraph::setTreeRoot(VarNode *root) {
        _tree_root = root;
    }

    VarNode *FactorGraph::treeRoot() {
        return _tree_root;
    }

    int FactorGraph::nHiddenVar() const {
        return (int) std::count_if(_vars.begin(), _vars.end(), [](const std::unique_ptr<VarNode> & elem) {
            return (elem->type() == VarNodeType::HIDDEN);
        });
    }

    int FactorGraph::nObservedVar() const {
        return (int) std::count_if(_vars.begin(), _vars.end(), [](const std::unique_ptr<VarNode> & elem) {
            return (elem->type() == VarNodeType::OBSERVED);
        });
    }

    VarNode *FactorGraph::node(int i) {
        return _vars[i].get();
    }

    int FactorGraph::nodes() const {
        return (int) _vars.size();
    }

    int FactorGraph::factors() const {
        return (int) _factors.size();
    }

    nodes::FactorNode *FactorGraph::factor(int i) {
        return _factors[i].get();
    }

    void FactorGraph::loadEvidence(int nobs, const std::string& file_name) {
        std::ifstream input(file_name);
        std::string line;
        std::string name;
        int obs;

        while (getline(input, line)) {
            unsigned long i = line.find(' ');
            if (i == std::string::npos) {
                throw std::runtime_error("Invalid file format: '" + file_name + "'");
            }
            name = line.substr(0, i);
            obs = std::stoi(line.substr(i + 1));
            ObservedVarIter it(this);
            while (*it != nullptr) {
                if ((*it)->name() == name) {
                    (*it)->setPosterior(Categorical::create(Ops::one_hot(nobs, obs)));
                }
                ++it;
            }
        }
    }

    void FactorGraph::removeHiddenChildren(VarNode *node) {
        for (auto it = node->firstChild(); it != node->lastChild() ; ++it) {
            auto child = (*it)->child();
            if (child->type() == VarNodeType::OBSERVED) {
                continue;
            } else {
                removeBranch(*it);
            }
        }
    }

    void FactorGraph::integrate(
            int action,
            const Tensor& observation,
            const Tensor& A,
            const Tensor& B
    ) {
        assert(B.dim() == 3 && "FactorGraph::integrate, B must be 3-tensor.");

        // Create a categorical distribution over action
        auto n_actions = B.size(2);
        Tensor action_param = API::full({n_actions}, 0.1 / ((double) n_actions - 1));
        action_param[action] = 0.9;
        auto a = API::Categorical(action_param);

        // Call generic integrate function
        integrate(a, observation, A, B);
    }

    void FactorGraph::integrate(
            int action,
            const Tensor& observation,
            VarNode *A,
            VarNode *B
    ) {
        assert(B->prior()->params().dim() == 3 && "FactorGraph::integrate, B must be 3-random-tensor.");

        // Create a categorical distribution over action
        auto B_param = B->prior()->params();
        long actions = B_param.size(B_param.dim() - 1);
        Tensor action_param = API::full({actions}, 0.1 / ((double) actions - 1));
        action_param[action] = 0.9;
        auto a = API::Categorical(action_param);

        // Call generic integrate function
        integrate(a, observation, A, B);
    }

    void FactorGraph::integrate(
            int action,
            const Tensor &observation,
            const std::shared_ptr<Tensor> &A,
            const std::shared_ptr<Tensor> &B
    ) {
        assert(B->dim() == 3 && "FactorGraph::integrate, B must be 3-tensor.");

        // Create a categorical distribution over action
        auto n_actions = B->size(2);
        Tensor action_param = API::full({n_actions}, 0.1 / ((double) n_actions - 1));
        action_param[action] = 0.9;
        auto a = API::Categorical(action_param);

        // Call generic integrate function
        integrate(a, observation, A, B);
    }

    void FactorGraph::integrate(
            VarNode *U,
            int action,
            const Tensor &observation,
            VarNode *A,
            VarNode *B
    ) {
        assert(U->prior()->type() == DIRICHLET && "FactorGraph::integrate, U must be distributed according to a Dirichlet.");
        assert(B->prior()->params().dim() == 3 && "FactorGraph::integrate, B must be 3-random-tensor.");

        // Increase Dirichlet parameters
        auto p = U->prior()->params();
        auto p_a = p.accessor<double,1>();
        p_a[action] += 1;
        U->prior()->updateParams(p);

        // Create a categorical distribution over action
        auto a = API::Categorical(U);

        // Call generic integrate function
        integrate(a, observation, A, B);
    }

    template<class T1, class T2>
    void FactorGraph::integrate(
            VarNode *a,
            const Tensor& observation,
            T1 A, T2 B
    ) {
        // Cut-off child branches
        removeHiddenChildren(_tree_root);

        // Create new slide of action/state/observation.
        auto *new_root = API::ActiveTransition(_tree_root, a, B);
        auto o         = API::Transition(new_root, A);
        o->setPosterior(Categorical::create(observation));
        o->setType(VarNodeType::OBSERVED);

        // Clean up the factor graph
        _tree_root->removeNullChildren();
        setTreeRoot(new_root);
    }

    void FactorGraph::removeBranch(FactorNode *node) {
        node->parent(0)->disconnectChild(node);
        auto itf = std::find_if(_factors.begin(), _factors.end(), [node](const std::unique_ptr<FactorNode>& n) {
            return n.get() == node;
        });
        auto itv = std::find_if(_vars.begin(), _vars.end(), [node](const std::unique_ptr<VarNode>& n) {
            return n.get() == node->child();
        });

        for (auto i = (*itv)->firstChild(); i != (*itv)->lastChild(); ++i) {
            removeBranch(*i);
        }
        itf->reset();
        itv->reset();
        removeNullNodes();
        removeNullFactors();
    }

    void FactorGraph::removeNullNodes() {
        _vars.erase(std::remove_if(_vars.begin(), _vars.end(),
                                   [](std::unique_ptr<VarNode> &x){return x == nullptr;}), _vars.end());
    }

    void FactorGraph::removeNullFactors() {
        _factors.erase(std::remove_if(_factors.begin(), _factors.end(),
                                      [](std::unique_ptr<FactorNode> &x){return x == nullptr;}), _factors.end());
    }

    std::vector<VarNode*> FactorGraph::getNodes() {
        std::vector<VarNode*> vars;

        for (auto & _var : _vars) {
            vars.push_back(_var.get());
        }
        return vars;
    }

    std::string FactorGraph::getName(const std::string &name, std::pair<std::string, int> &default_name) {
        std::string res;

        if (name.empty()) {
            res = default_name.first + std::to_string(default_name.second);
            ++default_name.second;
        } else {
            res = name;
        }
        return res;
    }

    void FactorGraph::writeGraphviz(const std::string &file_name, const std::vector<VarNodeAttr> &display) {
        static std::pair<std::string, int> dvn("n", 0);
        static std::pair<std::string, int> dfn("f", 0);
        std::ofstream file;
        file.open(file_name);

        file << "digraph G {\n";
        writeGraphvizNodes(file, dvn, dfn);
        writeGraphvizFactors(file, dvn, dfn);
        writeGraphvizData(file, display);
        file << "}\n";
        file.close();
    }

    void FactorGraph::writeGraphvizNodes(
            std::ofstream &file,
            std::pair<std::string,int> &dvn,
            std::pair<std::string,int> &dfn
    ) {
        for (auto & _var : _vars) {
            // Get/set nodes' names
            std::string parent_name = getName(_var->parent()->name(), dfn);
            _var->parent()->setName(parent_name);
            std::string var_name = getName(_var->name(), dvn);
            _var->setName(var_name);
            // Add "parent_name -> var_name" to the file
            file << "\t" << parent_name << " -> " << var_name << "\n";
            // If _var is observed set a gray background
            if (_var->type() == VarNodeType::OBSERVED) {
                file << "\t" << var_name << " [fillcolor=\"lightgrey\",style=filled]\n";
            }
        }
    }

    void FactorGraph::writeGraphvizFactors(
            std::ofstream &file,
            std::pair<std::string,int> &dvn,
            std::pair<std::string,int> &dfn
    ) {
        for (auto & _factor : _factors) {
            // Get/set nodes' names
            std::string factor_name = getName(_factor->name(), dfn);
            _factor->setName(factor_name);
            for (int j = 0; _factor->parent(j) != nullptr; ++j) {
                // Get/set nodes' names
                std::string parent_name = getName(_factor->parent(j)->name(), dvn);
                _factor->parent(j)->setName(parent_name);
                // Add "parent_name -> factor_name" to the file
                file << "\t" << parent_name << " -> " << factor_name << "[dir=none]\n";
            }
            // Make sure factors are displayed as squares
            file << "\t" << factor_name << " [shape=square]\n";
        }
    }

    void FactorGraph::writeGraphvizData(std::ofstream &file, const std::vector<VarNodeAttr> &display) {
        std::vector<std::string (*)(VarNode*)> func{
                [](VarNode *var){ return std::to_string(var->n()); },
                [](VarNode *var){ return (var->g() == std::numeric_limits<double>::min()) ? "-Infinity" : std::to_string(var->g()); },
                [](VarNode *var){ return std::to_string(var->action()); }
        };

        if (display.empty())
            return;
        for (auto & _var : _vars) {
            // Add "parent_name -> var_name" to the file
            file << "\t" << _var->name() << "_data -> " << _var->name() << " [dir=none,style=dashed,color=\"gray\"]\n";
            // Create label and display the data node
            std::string label = R"(<<table border="0" cellborder="1" cellspacing="0" cellpadding="4">)";
            for (auto i : display) {
                label += "<TR><TD bgcolor=\"YellowGreen\">" + attrNames[i] + "</TD>" +\
                             "<TD bgcolor=\"YellowGreen\">" + func[i](_var.get()) + "</TD></TR>";
            }
            label += "</table>>";
            file << "\t" << _var->name() << "_data [shape=none,margin=0,label=" << label << "]\n";
        }
    }

}