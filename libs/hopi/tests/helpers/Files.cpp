//
// Created by Theophile Champion on 03/12/2020.
//

#include "Files.h"

namespace tests {

    std::string Files::getMazePath(const std::string& file_name) {
        return "../examples/mazes/" + file_name;
    }

    std::string Files::getEvidencePath(const std::string &file_name) {
        return "../examples/evidences/" + file_name;
    }

}
