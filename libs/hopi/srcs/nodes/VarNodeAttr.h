//
// Created by tmac3 on 07/12/2020.
//

#ifndef HOMING_PIGEON_2_VARNODEATTR_H
#define HOMING_PIGEON_2_VARNODEATTR_H

namespace hopi::nodes {

    enum VarNodeAttr : int {
        N = 0,
        G = 1,
        A = 2
    };

    static const std::string attrNames[] = { "N", "G", "U" };

}

#endif //HOMING_PIGEON_2_VARNODEATTR_H
