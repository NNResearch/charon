//
// Created by shankara on 9/13/18.
//

#ifndef PROJECT_UTILS_H
#define PROJECT_UTILS_H

#include <zonotope.h>
#include "eigen_wrapper.hpp"
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>

class interval {
    public:
        Vec lower;
        Vec upper;
        std::vector<int> posDims;

        interval();
        interval(Vec, Vec);

        elina_interval_t** get_elina_interval() const;
        void set_bounds(Vec, Vec);
        Vec get_center() const;
};

interval::interval(Vec l, Vec u): lower(l), upper(u) {
    for(int i = 0; i < l.size(); i++) {
        if (upper(i) - lower(i) > 0)
            this->posDims.push_back(i);
    }
}

interval::interval(): lower(Vec(0)), upper(Vec(0)) {}

void interval::set_bounds(Vec l, Vec u) {
    lower = l;
    upper = u;
}

Vec interval::get_center() const {
    Vec ce(lower.size());
    for (int i = 0; i < lower.size(); i++) {
        double l = lower(i);
        double u = upper(i);
        ce(i) = (l + u) / 2.0;
    }
    return ce;
}

interval read_property1(std::string filename, int& dims) {
    std::vector<double> lower;
    std::vector<double> upper;
    std::ifstream in(filename.c_str());
    std::string line;
    while (getline(in, line)) {
        std::istringstream iss(line);
        double l, u;
        iss.get();
        iss >> l;
        iss.get();
        iss.get();
        iss >> u;
        lower.push_back(l);
        upper.push_back(u);
    }

    dims = lower.size();
    Vec l(dims);
    Vec u(dims);
    for (int i = 0; i < dims; i++) {
        l(i) = lower[i];
        u(i) = upper[i];
    }

    return interval(l, u);
}


#endif //PROJECT_UTILS_H
