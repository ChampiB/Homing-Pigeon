//
// Created by Theophile Champion on 25/06/2021.
//

#ifndef HOMING_PIGEON_GRAPHVIZ_H
#define HOMING_PIGEON_GRAPHVIZ_H

#include <string>
#include <memory>
#include <vector>
#include <fstream>
#include "nodes/VarNodeAttr.h"

namespace hopi::nodes {
    class VarNode;
    class FactorNode;
}

namespace hopi::graphs {

    class GraphViz {
    public:
        /**
         * Constructor. Open the output and write graph opening statement into it.
         * @param file_name output file name
         */
        explicit GraphViz(const std::string &file_name);

        /**
         * Destructor. Write the closing statement in the output file and close the output file.
         */
        ~GraphViz();

        /**
         * Write all the variable nodes in the file in the Graphviz format
         * @param dvn the default name for variable nodes
         * @param dfn the default name for factor nodes
         * @param vars the variable nodes
         */
        void writeNodes(
                std::pair<std::string,int> &dvn,
                std::pair<std::string,int> &dfn,
                const std::vector<std::unique_ptr<nodes::VarNode>> &vars
        );

        /**
         * Write all the factor nodes in the file in the Graphviz format
         * @param dvn the default name for variable nodes
         * @param dfn the default name for factor nodes
         * @param factors the factor nodes
         */
        void writeFactors(
                std::pair<std::string,int> &dvn,
                std::pair<std::string,int> &dfn,
                const std::vector<std::unique_ptr<nodes::FactorNode>> &factors
        );

        /**
         * Write the nodes' attributes in the file using the Graphviz format.
         * @param dvn the default name for variable nodes
         * @param vars the variable nodes
         * @param display the list of attributes to that must be displayed, i.e., witten in the file
         */
        void writeData(
                std::pair<std::string,int> &dvn,
                const std::vector<std::unique_ptr<nodes::VarNode>> &vars,
                const std::vector<nodes::VarNodeAttr> &display
        );

        /**
         * Generate chart of the distribution in the graph and add them to the file using the Graphviz format.
         * @param dvn the default name for variable nodes
         * @param vars the variable nodes
         */
        void writePosteriors(
                std::pair<std::string,int> &dvn,
                const std::vector<std::unique_ptr<nodes::VarNode>> &vars
        );

    private:
        /**
         * Getter.
         * @param name the node's name
         * @param default_name the default name to be used if "name" is empty
         * @return "name" if not empty, default name otherwise
         */
        static std::string getName(const std::string &name, std::pair<std::string, int> &default_name);

    private:
        std::string _file_name;
        std::ofstream _file;
    };

}

#endif //HOMING_PIGEON_GRAPHVIZ_H
