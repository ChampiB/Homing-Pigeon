//
// Created by Theophile Champion on 25/06/2021.
//

#include "GraphViz.h"
#include "nodes/VarNode.h"
#include "nodes/FactorNode.h"
#include "distributions/Distribution.h"
#include "gnuplot-iostream.h"
#include "algorithms/planning/MCTSNodeData.h"

using namespace hopi::nodes;

namespace hopi::graphs {

    GraphViz::GraphViz(const std::string &file_name) {
        _file_name = file_name;
        _file.open(file_name);
        _file << "digraph G {\n";
    }

    GraphViz::~GraphViz() {
        _file << "}\n";
        _file.close();
    }

    void GraphViz::writeNodes(
            std::pair<std::string, int> &dvn,
            std::pair<std::string, int> &dfn,
            const std::vector<std::unique_ptr<VarNode>> &vars
    ) {
        for (auto & _var : vars) {
            // Get/set nodes' names
            std::string parent_name = getName(_var->parent()->name(), dfn);
            _var->parent()->setName(parent_name);
            std::string var_name = getName(_var->name(), dvn);
            _var->setName(var_name);
            // Add "parent_name -> var_name" to the file
            _file << "\t" << parent_name << " -> " << var_name << "\n";
            // If _var is observed set a gray background
            if (_var->type() == VarNodeType::OBSERVED) {
                _file << "\t" << var_name << " [fillcolor=\"lightgrey\",style=filled]\n";
            }
        }
    }

    void GraphViz::writeFactors(
            std::pair<std::string, int> &dvn,
            std::pair<std::string, int> &dfn,
            const std::vector<std::unique_ptr<FactorNode>> &factors
    ) {
        for (auto & _factor : factors) {
            // Get/set nodes' names
            std::string factor_name = getName(_factor->name(), dfn);
            _factor->setName(factor_name);
            for (int j = 0; _factor->parent(j) != nullptr; ++j) {
                // Get/set nodes' names
                std::string parent_name = getName(_factor->parent(j)->name(), dvn);
                _factor->parent(j)->setName(parent_name);
                // Add "parent_name -> factor_name" to the file
                _file << "\t" << parent_name << " -> " << factor_name << "[dir=none]\n";
            }
            // Make sure factors are displayed as squares
            _file << "\t" << factor_name << " [shape=square]\n";
        }
    }

    void GraphViz::writeData(
            std::pair<std::string, int> &dvn,
            const std::vector<std::unique_ptr<VarNode>> &vars,
            const std::vector<VarNodeAttr> &display
    ) {
        std::vector<std::string (*)(VarNode*)> func{
                [](VarNode *var){ return std::to_string(var->data()->visits); },
                [](VarNode *var){ return (var->data()->cost == std::numeric_limits<double>::min()) ? "-Infinity" : std::to_string(var->data()->cost); },
                [](VarNode *var){ return std::to_string(var->data()->action); },
                [](VarNode *var){ return std::to_string(var->data()->pruned); },
                [](VarNode *var){ return std::to_string(argmax(var->posterior()->params()).item<int>()); }
        };

        if (display.empty())
            return;
        for (auto & var : vars) {
            std::string var_name = getName(var->name(), dvn);
            // Add "parent_name -> var_name" to the file
            _file << "\t" << var_name << "_data -> " << var_name << " [dir=none,style=dashed,color=\"gray\"]\n";
            // Create label and display the data node
            std::string label = R"(<<table border="0" cellborder="1" cellspacing="0" cellpadding="4">)";
            for (auto i : display) {
                label += "<TR><TD bgcolor=\"YellowGreen\">" + attrNames[i] + "</TD>" + \
                             "<TD bgcolor=\"YellowGreen\">" + func[i](var.get()) + "</TD></TR>";
            }
            label += "</table>>";
            _file << "\t" << var_name << "_data [shape=none,margin=0,label=" << label << "]\n";
        }
    }

    void GraphViz::writePosteriors(
            std::pair<std::string,int> &dvn,
            const std::vector<std::unique_ptr<nodes::VarNode>> &vars
    ) {
        for (int i = 0; i < vars.size(); ++i) {
            // Create the file describing the posterior distribution
            if (vars[i]->posterior()->type() != distributions::CATEGORICAL)
                continue;
            auto param = vars[i]->posterior()->params();
            std::string file_distrib_name = _file_name + ".distrib";
            std::ofstream file_distrib;
            file_distrib.open(file_distrib_name);
            for (int j = 0; j < param.size(0); ++j) {
                file_distrib << std::to_string(j) << " " << std::to_string(param[j].item<double>()) << "\n";
            }
            file_distrib.close();

            // Create the chart of the distribution
            std::string file_chart_name = _file_name + "." + std::to_string(i) + ".chart.png";
            Gnuplot gp("gnuplot");
            gp << "reset\n";
            gp << "set style data histograms\n";
            gp << "set yrange [0 : *]\n";
            gp << "set offsets graph 0, 0, 0.05, 0\n";
            gp << "set ylabel \"P(X)\"\n";
            gp << "set xlabel \"X\"\n";
            gp << "set terminal png size 550,200\n";
            gp << "set tmargin 0.5\n";
            gp << "set bmargin 3\n";
            gp << "set output '" + file_chart_name + "'\n";
            gp << "plot \"" + file_distrib_name + "\" using 2:xtic(1) notitle\n";

            // Add the chart to the graphviz output
            std::string var_name = getName(vars[i]->name(), dvn);
            _file << "\t" << var_name << R"(_image [label="",shape=box,image=")" + file_chart_name + "\"]\n";
            _file << "\t" << var_name << "_image -> " + var_name + "\n";
        }
    }

    std::string GraphViz::getName(const std::string &name, std::pair<std::string, int> &default_name) {
        std::string res;

        if (name.empty()) {
            res = default_name.first + std::to_string(default_name.second);
            ++default_name.second;
        } else {
            res = name;
        }
        return res;
    }

}
