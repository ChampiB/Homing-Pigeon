//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_FACTORGRAPH_H
#define HOMING_PIGEON_2_FACTORGRAPH_H

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

    class FactorGraph {
    public:
        static std::shared_ptr<FactorGraph> current();
        static void setCurrent(std::shared_ptr<FactorGraph> &ptr);
        static void setCurrent(std::shared_ptr<FactorGraph> &&ptr);

    public:
        FactorGraph();
        void integrate(
                nodes::VarNode *U,
                int action,
                const torch::Tensor& observation,
                nodes::VarNode *A,
                nodes::VarNode *B
        );
        void integrate(
                int action,
                const torch::Tensor& observation,
                const torch::Tensor& A,
                const torch::Tensor& B
        );
        void integrate(
                int action,
                const torch::Tensor& observation,
                nodes::VarNode *A,
                nodes::VarNode *B
        );
        void setTreeRoot(nodes::VarNode *root);
        nodes::VarNode *treeRoot();
        nodes::VarNode *addNode(std::unique_ptr<nodes::VarNode> node);
        nodes::FactorNode *addFactor(std::unique_ptr<nodes::FactorNode> node);
        [[nodiscard]] int nHiddenVar() const;
        [[nodiscard]] int nObservedVar() const;
        [[nodiscard]] int nodes() const;
        std::vector<nodes::VarNode*> getNodes();
        [[nodiscard]] int factors() const;
        nodes::VarNode *node(int index);
        nodes::FactorNode *factor(int index);
        void loadEvidence(int nobs, const std::string& file_name);
        void removeHiddenChildren(nodes::VarNode *node);
        void removeBranch(nodes::FactorNode *node);
        void removeNullNodes();
        void removeNullFactors();
        void writeGraphviz(const std::string& file_name, const std::vector<nodes::VarNodeAttr> &display);

    private:
        void writeGraphvizNodes  (std::ofstream &file, std::pair<std::string,int> &dvn, std::pair<std::string,int> &dfn);
        void writeGraphvizFactors(std::ofstream &file, std::pair<std::string,int> &dvn, std::pair<std::string,int> &dfn);
        void writeGraphvizData   (std::ofstream &file, const std::vector<nodes::VarNodeAttr> &display);
        static std::string getName(const std::string &name, std::pair<std::string, int> &default_name);

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

#endif //HOMING_PIGEON_2_FACTORGRAPH_H
