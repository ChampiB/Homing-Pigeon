//
// Created by tmac3 on 28/11/2020.
//

#include <fstream>
#include <distributions/Dirichlet.h>
#include "distributions/Categorical.h"
#include "nodes/VarNode.h"
#include "nodes/FactorNode.h"
#include "math/Functions.h"
#include "api/API.h"
#include "FactorGraph.h"
#include "iterators/ObservedVarIter.h"

using namespace hopi::nodes;
using namespace hopi::iterators;
using namespace hopi::distributions;
using namespace hopi::math;
using namespace hopi::api;
using namespace Eigen;

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

    FactorNode *FactorGraph::addFactor(std::unique_ptr<FactorNode> node) {
        _factors.push_back(std::move(node));
        return _factors[_factors.size() - 1].get();
    }

    void FactorGraph::setTreeRoot(VarNode *root) {
        _tree_root = root;
    }

    VarNode *FactorGraph::treeRoot() {
        return _tree_root;
    }

    int FactorGraph::nHiddenVar() const {
        return std::count_if(_vars.begin(), _vars.end(), [](const std::unique_ptr<VarNode> & elem) {
            return (elem->type() == VarNodeType::HIDDEN);
        });
    }

    int FactorGraph::nObservedVar() const {
        return std::count_if(_vars.begin(), _vars.end(), [](const std::unique_ptr<VarNode> & elem) {
            return (elem->type() == VarNodeType::OBSERVED);
        });
    }

    VarNode *FactorGraph::node(int index) {
        return _vars[index].get();
    }

    int FactorGraph::nodes() const {
        return _vars.size();
    }

    int FactorGraph::factors() const {
        return _factors.size();
    }

    nodes::FactorNode *FactorGraph::factor(int index) {
        return _factors[index].get();
    }

    void FactorGraph::loadEvidence(int nobs, const std::string& file_name) {
        std::ifstream input(file_name);
        std::string line;
        std::string name;
        int obs;

        while (getline(input, line)) {
            int i = line.find(' ');
            if (i == std::string::npos) {
                throw std::runtime_error("Invalid '.evi' file: no space between name and observation.");
            }
            name = line.substr(0, i);
            obs = std::stoi(line.substr(i + 1));
            ObservedVarIter it(this);
            while (*it != nullptr) {
                if ((*it)->name() == name) {
                    (*it)->setPosterior(Categorical::create(Functions::oneHot(nobs, obs)));
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
            const MatrixXd& observation,
            const MatrixXd& A,
            const std::vector<MatrixXd>& B
    ) {
        removeHiddenChildren(_tree_root);
        MatrixXd action_param = MatrixXd::Constant(B.size(), 1, 0.1 / (B.size() - 1));
        action_param(action, 0) = 0.9;
        auto a = API::Categorical(action_param);
        integrate(a, observation, A, B);
    }

    void FactorGraph::integrate(
            int action,
            const MatrixXd& observation,
            VarNode *A,
            VarNode *B
    ) {
        removeHiddenChildren(_tree_root);
        int actions = B->prior()->params().size();
        MatrixXd action_param = MatrixXd::Constant(actions, 1, 0.1 / (actions - 1));
        action_param(action, 0) = 0.9;
        auto a = API::Categorical(action_param);
        integrate(a, observation, A, B);
    }

    void FactorGraph::integrate(
            VarNode *U,
            int action,
            const MatrixXd &observation,
            VarNode *A,
            VarNode *B
    ) {
        // Check that U is a Dirichlet node.
        if (U->prior()->type() != DIRICHLET) {
            throw std::runtime_error("Integrate(U, action, observation, A, B) assumes U is a Dirichlet.");
        }
        // Create new slide of action/state/observation.
        auto d = dynamic_cast<Dirichlet*>(U->prior());
        d->increaseParam(0, action, 0);
        auto a = API::Categorical(U);
        integrate(a, observation, A, B);
    }

    template<class T1, class T2>
    void FactorGraph::integrate(
            VarNode *a,
            const Eigen::MatrixXd& observation,
            T1 A, T2 B
    ) {
        // Create new slide of action/state/observation.
        auto *new_root = API::ActiveTransition(_tree_root, a, B);
        auto o         = API::Transition(new_root, A);
        o->setPosterior(Categorical::create(observation));
        o->setType(VarNodeType::OBSERVED);
        // Clean up the factor graph
        _tree_root->removeNullChildren();
        removeNullNodes();
        removeNullFactors();
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
    }

    void FactorGraph::removeNullNodes() {
        std::vector<std::unique_ptr<VarNode>>::iterator it;

        while ((it = std::find(_vars.begin(), _vars.end(), nullptr)) != _vars.end()) {
            _vars.erase(it);
        }
    }

    void FactorGraph::removeNullFactors() {
        std::vector<std::unique_ptr<FactorNode>>::iterator it;

        while ((it = std::find(_factors.begin(), _factors.end(), nullptr)) != _factors.end()) {
            _factors.erase(it);
        }
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