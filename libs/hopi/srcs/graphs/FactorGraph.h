//
// Created by Theophile Champion on 28/11/2020.
//

#ifndef HOMING_PIGEON_FACTOR_GRAPH_H
#define HOMING_PIGEON_FACTOR_GRAPH_H

#include <memory>
#include <vector>
#include <torch/torch.h>
#include "nodes/VarNodeType.h"
#include "nodes/VarNodeAttr.h"

namespace hopi::nodes {
    class VarNode;
    class FactorNode;
}

namespace hopi::graphs {

    /**
     * The class representing a factor graph.
     */
    class FactorGraph {
    public:
        //
        // The current factor graph on which the user works is stored as a static variable. The following functions
        // allows the user to access this factor graph and to change the factor graph on which he (i.e., the user)
        // is working.
        //

        /**
         * Getter.
         * @return the current factor graph
         */
        static std::shared_ptr<FactorGraph> current();

        /**
         * Replace the current factor graph by the one sent as parameters. If nullptr is sent as input, then the
         * framework is reset to its initial state, and the next call to FactorGraph::Current() will create a
         * completely new graph.
         * @param ptr a pointer to the new factor graph
         */
        static void setCurrent(std::shared_ptr<FactorGraph> &ptr);

        /**
         * Replace the current factor graph by the one sent as parameters. If nullptr is sent as input, then the
         * framework is reset to its initial state, and the next call to FactorGraph::Current() will create a
         * completely new graph.
         * @param ptr a pointer to the new factor graph
         */
        static void setCurrent(std::shared_ptr<FactorGraph> &&ptr);

    public:
        //
        // Constructor
        //
        /**
         * Create a new factor graph without any nodes.
         */
        FactorGraph();

        /**
         * Cut-off the branches of the tree that was expanded during planning, then add a new slice to the POMDP by
         * assuming that the action "action" has been taken and that the observation "observation" has been made.
         * @param U the parameter of the prior over actions
         * @param action the action performed
         * @param observation the observation made
         * @param A the likelihood mapping to use when adding the slice
         * @param B the transition mapping to use when adding the slice
         */
        void integrate(
                nodes::VarNode *U,
                int action,
                const torch::Tensor& observation,
                nodes::VarNode *A,
                nodes::VarNode *B
        );

        /**
         * Cut-off the branches of the tree that was expanded during planning, then add a new slice to the POMDP by
         * assuming that the action "action" has been taken and that the observation "observation" has been made.
         * @param action the action performed
         * @param observation the observation made
         * @param A the likelihood mapping to use when adding the slice
         * @param B the transition mapping to use when adding the slice
         */
        void integrate(
                int action,
                const torch::Tensor& observation,
                const torch::Tensor& A,
                const torch::Tensor& B
        );

        /**
         * Cut-off the branches of the tree that was expanded during planning, then add a new slice to the POMDP by
         * assuming that the action "action" has been taken and that the observation "observation" has been made.
         * @param action the action performed
         * @param observation the observation made
         * @param A the likelihood mapping to use when adding the slice
         * @param B the transition mapping to use when adding the slice
         */
        void integrate(
                int action,
                const torch::Tensor& observation,
                nodes::VarNode *A,
                nodes::VarNode *B
        );

        /**
         * Setter.
         * @param root the new root of the tree
         */
        void setTreeRoot(nodes::VarNode *root);

        /**
         * Getter.
         * @return the tree of the root
         */
        nodes::VarNode *treeRoot();

        /**
         * Add a new variable node to the graph
         * @param node the node to be added
         * @return the added node
         */
        nodes::VarNode *addNode(std::unique_ptr<nodes::VarNode> node);

        /**
         * Getter.
         * @return the list of all nodes
         */
        std::vector<nodes::VarNode*> getNodes();

        /**
         * Getter.
         * @param i the index of the node that needs to be access
         * @return the i-th node
         */
        nodes::VarNode *node(int i);

        /**
         * Add a factor to the graph.
         * @param factor the factor be added
         * @return the added factor
         */
        nodes::FactorNode *addFactor(std::unique_ptr<nodes::FactorNode> factor);

        /**
         * Getter.
         * @param i the index of the factor that needs to be accessed
         * @return the i-th factor
         */
        nodes::FactorNode *factor(int i);

        /**
         * Getter.
         * @return the number of hidden variables in the graph
         */
        [[nodiscard]] int nHiddenVar() const;

        /**
         * Getter.
         * @return the number of observed variables in the graph.
         */
        [[nodiscard]] int nObservedVar() const;

        /**
         * Getter.
         * @return the number of variables in the graph.
         */
        [[nodiscard]] int nodes() const;

        /**
         * Getter.
         * @return the number of factors in the graph.
         */
        [[nodiscard]] int factors() const;

        /**
         * Load the evidence from a file into the factor graph.
         * @param nobs the number of observations
         * @param file_name the name of the evidence file
         */
        void loadEvidence(int nobs, const std::string& file_name);

        /**
         * Write the factor graph in a file using the Graphviz format.
         * @param file_name the name of the output file
         * @param display the attributes that must be written in the output file
         */
        void writeGraphviz(const std::string& file_name, const std::vector<nodes::VarNodeAttr> &display);

        /**
         * Cut-off the branch corresponding to the input node
         * @param node the node at the top of the branch to be deleted
         */
        void removeBranch(nodes::FactorNode *node);

    private:
        /**
         * Remove all the nullptr from the list of variable nodes.
         */
        void removeNullNodes();

        /**
         * Remove all the nullptr from the list of factor nodes.
         */
        void removeNullFactors();

        /**
         * Remove all the node's children which are not observed variables.
         * @param node the node whose hidden children must be removed
         */
        void removeHiddenChildren(nodes::VarNode *node);

        /**
         * Write all the variable nodes in the file in the Graphviz format
         * @param file the output file
         * @param dvn the default name for variable nodes
         * @param dfn the default name for factor nodes
         */
        void writeGraphvizNodes(std::ofstream &file, std::pair<std::string,int> &dvn, std::pair<std::string,int> &dfn);

        /**
         * Write all the factor nodes in the file in the Graphviz format
         * @param file the output file
         * @param dvn the default name for variable nodes
         * @param dfn the default name for factor nodes
         */
        void writeGraphvizFactors(std::ofstream &file, std::pair<std::string,int> &dvn, std::pair<std::string,int> &dfn);

        /**
         * Write the nodes' attributes in the file using the Graphviz format.
         * @param file the output file
         * @param display the list of attributes to that must be displayed, i.e., witten in the file
         */
        void writeGraphvizData(std::ofstream &file, const std::vector<nodes::VarNodeAttr> &display);

        /**
         * Getter.
         * @param name the node's name
         * @param default_name the default name to be used if "name" is empty
         * @return "name" if not empty, default name otherwise
         */
        static std::string getName(const std::string &name, std::pair<std::string, int> &default_name);

        /**
         * Cut-off the branches of the tree that was expanded during planning, then add a new slice to the POMDP by
         * assuming that the action "a" has been taken and that the observation "observation" has been made.
         * @tparam T1 the type of the likelihood mapping
         * @tparam T2 the type of the transition mapping
         * @param a the action performed
         * @param observation the observation made
         * @param A the likelihood mapping
         * @param B the transition mapping
         */
        template<class T1, class T2>
        void integrate(
                nodes::VarNode *a,
                const torch::Tensor& observation,
                T1 A, T2 B
        );

    private:
        std::vector<std::unique_ptr<nodes::VarNode>> _vars;
        std::vector<std::unique_ptr<nodes::FactorNode>> _factors;
        nodes::VarNode *_tree_root;
    };

}

#endif //HOMING_PIGEON_FACTOR_GRAPH_H
