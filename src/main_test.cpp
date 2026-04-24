#include <iostream>
#include "LinearModel.hpp"

int main() {

    // TEST 1 : lineairement separable
    std::vector<double> X = {
         1.0,  1.0,
         2.0,  2.0,
        -1.0, -1.0,
        -2.0, -2.0
    };
    std::vector<double> Y = {1.0, 1.0, -1.0, -1.0};

    LinearModel model(2);
    model.train(X, Y, 0.1, 50);

    std::cout << "=== TEST 1 : Lineairement separable ===\n";
    int correct_1 = 0;
    for (int i = 0; i < 4; i++) {
        std::vector<double> sample = {X[i*2], X[i*2+1]};
        double pred = model.predict(sample);
        bool ok = (pred == Y[i]);
        if (ok) correct_1++;
        std::cout << "  pred=" << pred << "  attendu=" << Y[i]
                  << (ok ? "  OK" : "  ERREUR") << "\n";
    }
    std::cout << "  Accuracy : " << correct_1 << "/4 (attendu : 4/4)\n\n";

    // TEST 2 : XOR sans transformation
    std::vector<double> X_xor = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    };
    std::vector<double> Y_xor = {-1.0, 1.0, 1.0, -1.0};

    LinearModel model_xor(2);
    model_xor.train(X_xor, Y_xor, 0.1, 100);

    std::cout << "=== TEST 2 : XOR (echec attendu) ===\n";
    int correct_2 = 0;
    for (int i = 0; i < 4; i++) {
        std::vector<double> sample = {X_xor[i*2], X_xor[i*2+1]};
        double pred = model_xor.predict(sample);
        bool ok = (pred == Y_xor[i]);
        if (ok) correct_2++;
        std::cout << "  pred=" << pred << "  attendu=" << Y_xor[i]
                  << (ok ? "  OK" : "  ERREUR") << "\n";
    }
    std::cout << "  Accuracy : " << correct_2 << "/4 (attendu : < 4/4)\n\n";

    // TEST 3 : XOR avec transformation non-lineaire
    std::vector<double> X_xor_trans;
    for (int i = 0; i < 4; i++) {
        double x0 = X_xor[i*2];
        double x1 = X_xor[i*2+1];
        X_xor_trans.push_back(x0);
        X_xor_trans.push_back(x1);
        X_xor_trans.push_back(x0 * x1);
    }

    LinearModel model_trans(3);
    model_trans.train(X_xor_trans, Y_xor, 0.1, 500);

    std::cout << "=== TEST 3 : XOR avec transformation ===\n";
    int correct_3 = 0;
    for (int i = 0; i < 4; i++) {
        std::vector<double> sample = {X_xor_trans[i*3], X_xor_trans[i*3+1], X_xor_trans[i*3+2]};
        double pred = model_trans.predict(sample);
        bool ok = (pred == Y_xor[i]);
        if (ok) correct_3++;
        std::cout << "  pred=" << pred << "  attendu=" << Y_xor[i]
                  << (ok ? "  OK" : "  ERREUR") << "\n";
    }
    std::cout << "  Accuracy : " << correct_3 << "/4 (attendu : 4/4)\n\n";

    // TEST 4 : Regression lineaire y = 2x + 1
    // Le modele doit trouver weight ~ 2.0, bias ~ 1.0
    std::vector<double> X_reg = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> Y_reg = {3.0, 5.0, 7.0, 9.0, 11.0};

    LinearModel model_reg(1, LinearModel::REGRESSION);
    model_reg.train(X_reg, Y_reg, 0.01, 1000);

    std::cout << "=== TEST 4 : Regression lineaire (y = 2x + 1) ===\n";
    for (int i = 0; i < 5; i++) {
        std::vector<double> sample = {X_reg[i]};
        double pred = model_reg.predict(sample);
        std::cout << "  x=" << sample[0]
                  << "  pred=" << pred
                  << "  attendu=" << Y_reg[i] << "\n";
    }
    std::cout << "  (les predictions doivent etre proches des attendus)\n\n";

    // TEST 5 : Regression sur y = x2 (echec attendu)
    // Modele lineaire trop simple pour une courbe
    std::vector<double> X_quad = {-2.0, -1.0, 0.0, 1.0, 2.0};
    std::vector<double> Y_quad = {4.0, 1.0, 0.0, 1.0, 4.0};

    LinearModel model_quad(1, LinearModel::REGRESSION);
    model_quad.train(X_quad, Y_quad, 0.01, 1000);

    std::cout << "=== TEST 5 : Regression y=x2 (echec attendu) ===\n";
    for (int i = 0; i < 5; i++) {
        std::vector<double> sample = {X_quad[i]};
        double pred = model_quad.predict(sample);
        std::cout << "  x=" << sample[0]
                  << "  pred=" << pred
                  << "  attendu=" << Y_quad[i] << "\n";
    }
    std::cout << "  (predictions mauvaises -- modele trop simple)\n\n";

    // RESUME
    std::cout << "=== RESUME ===\n";
    std::cout << "  Separable          : " << correct_1 << "/4\n";
    std::cout << "  XOR sans transform : " << correct_2 << "/4\n";
    std::cout << "  XOR avec transform : " << correct_3 << "/4\n";

    return 0;
}