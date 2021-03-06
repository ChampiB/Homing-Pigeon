//
// Created by Theophile Champion on 03/12/2020.
//

#ifndef HOMING_PIGEON_FILES_H
#define HOMING_PIGEON_FILES_H

#include <string>

namespace tests {

    class Files {
    public:
        static std::string getMazePath(const std::string& file_name);
        static std::string getEvidencePath(const std::string& file_name);
    };

}

#endif //HOMING_PIGEON_FILES_H
