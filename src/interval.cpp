//
// Created by shankara on 10/12/18.
//

#include <zonotope.h>

#include "interval.hpp"

#include <fstream>

#include <iostream>
using namespace std;

Interval::Interval(Vec l, Vec u): lower{l}, upper{u} {
    if (l.size() != u.size()) {
        throw std::runtime_error("Length mismatch in Interval constructor");
    }
    for(int i = 0; i < l.size(); i++) {
        if (u(i) - l(i) > 0)
            this->posDims.push_back(i);
    }
}

Interval::Interval(): lower{Vec(0)}, upper{Vec(0)} {}

Interval::Interval(std::string filename) {
    cout << "Property file is: " << filename << "\n";
    std::vector<double> low;
    std::vector<double> upp;
    std::ifstream in(filename.c_str());
    std::string line;
    while (getline(in, line)) {
        std::istringstream iss(line);
        double l, u;
        iss.get(); // [
        iss >> l; // lower bound
        iss.get(); // ,
        iss.get(); // space
        iss >> u; // upper bound
        low.push_back(l);
        upp.push_back(u);
    }

    this->lower = Vec::Map(low.data(), low.size());
    this->upper = Vec::Map(upp.data(), upp.size());

    for (uint i = 0; i < low.size(); i++) {
        if (upp[i] - low[i] > 0) {
            this->posDims.push_back(i);
        }
    }
    cout << " lower vector: " << this->lower << "\n";
    cout << " upper vector: " << this->upper << "\n";
}

elina_interval_t** Interval::get_elina_interval() const {
    elina_interval_t** ret = (elina_interval_t**) malloc(this->lower.size() * sizeof(elina_interval_t*));
    for (int i = 0; i < lower.size(); i++) {
        ret[i] = elina_interval_alloc();
        elina_interval_set_double(ret[i], this->lower(i), this->upper(i));
    }
    return ret;
}

void Interval::set_bounds(Vec l, Vec u) {
    if (l.size() != u.size()) {
        throw std::runtime_error("Size mismatch in Interval::set_bounds");
    }
    this->lower = l;
    this->upper = u;
    posDims.clear();
    for (int i = 0; i < lower.size(); i++) {
        if (u(i) - l(i) > 0) {
            posDims.push_back(i);
        }
    }
}

Vec Interval::get_center() const {
    Vec ce(lower.size());
    for (int i = 0; i < lower.size(); i++) {
        double l = this->lower(i);
        double u = this->upper(i);
        ce(i) = (l + u) / 2.0;
    }
    return ce;
}

int Interval::longest_dim() const {
    int longest_dim = 0;
    double length = 0;
    for (int i = 0; i < this->lower.size(); i++) {
        if (this->upper(i) - this->lower(i) > length) {
            longest_dim = i;
            length = this->upper(i) - this->lower(i);
        }
    }
    return longest_dim;
}

int Interval::longest_dim(const Vec &counterexample) const {
    int longest_dim = 0;
    double length = 0;
    Vec center = this->get_center();
    for (int i = 0; i < this->lower.size(); i++) {
        double diff = std::abs(counterexample(i) - center(i));
        if (diff > length) {
            length = diff;
            longest_dim = i;
        }
    }
    return longest_dim;
}

double Interval::average_len() const {
    double avg = 0.0;
    for (int i = 0; i < this->lower.size(); i++) {
        avg += this->upper(i) - this->lower(i);
    }
    if (this->lower.size() > 0) {
        avg /= this->lower.size();
    }
    return avg;
}

