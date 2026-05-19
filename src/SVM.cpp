#include "SVM.h"
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

SVM::SVM(int input_size, double C, double epsilon, Mode mode) : bias(0.0), C(C), epsilon(epsilon), mode(mode){
    weights.resize(input_size);
    for (int i = 0; i < input_size; i++)
        weights[i] = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * 0.02 - 0.01;
}

double SVM::predict_raw(const std::vector<double> &x) const {
    double sum = bias;
    for (int i = 0; i < static_cast<int>(weights.size()); i++)
        sum += weights[i] * x[i];
    return sum;
}

double SVM::predict(const std::vector<double> &x) const {
    return predict_raw(x) >= 0.0 ? 1.0 : -1.0;
}

void SVM::train(const std::vector<std::vector<double> > &X, const std::vector<double> &Y, double learning_rate, int epochs) {

}

void SVM::save(const char *filename) {
    //TODO
}

void SVM::load(const char *filename) {
    //TODO
}

