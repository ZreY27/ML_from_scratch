#include <iostream>
#include "LinearModel.hpp"
#include "MLP.hpp"
#include "SVM.h"
#include "RBF.hpp"

int main() {
    system("chcp 65001 > nul");
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

    // TEST 2 : XOR sans transformation (echec attendu)
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
    std::vector<double> X_reg = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> Y_reg = {3.0, 5.0, 7.0, 9.0, 11.0};

    LinearModel model_reg(1, false);
    model_reg.train(X_reg, Y_reg, 0.01, 1000);

    std::cout << "=== TEST 4 : Regression lineaire (y = 2x + 1) ===\n";
    for (int i = 0; i < 5; i++) {
        std::vector<double> sample = {X_reg[i]};
        double pred = model_reg.predict(sample);
        std::cout << "  x=" << X_reg[i]
                  << "  pred=" << pred
                  << "  attendu=" << Y_reg[i] << "\n";
    }
    std::cout << "  (les predictions doivent etre proches des attendus)\n\n";

    // TEST 5 : Regression sur y = x2 (echec attendu)
    std::vector<double> X_quad = {-2.0, -1.0, 0.0, 1.0, 2.0};
    std::vector<double> Y_quad = {4.0, 1.0, 0.0, 1.0, 4.0};

    LinearModel model_quad(1, false);
    model_quad.train(X_quad, Y_quad, 0.01, 1000);

    std::cout << "=== TEST 5 : Regression y=x2 (echec attendu) ===\n";
    for (int i = 0; i < 5; i++) {
        std::vector<double> sample = {X_quad[i]};
        double pred = model_quad.predict(sample);
        std::cout << "  x=" << X_quad[i]
                  << "  pred=" << pred
                  << "  attendu=" << Y_quad[i] << "\n";
    }
    std::cout << "  (predictions mauvaises -- modele trop simple)\n\n";

    // ==========================================
    // TESTS MLP
    // ==========================================
    std::cout << "\n==========================================\n";
    std::cout << "               TESTS MLP\n";
    std::cout << "==========================================\n\n";

    std::vector<double> X_xor_mlp = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    };
    std::vector<double> Y_xor_mlp = {-1.0, 1.0, 1.0, -1.0};

    MLP mlp_classif({2, 3, 1}, true);
    mlp_classif.train(X_xor_mlp, Y_xor_mlp, 15000, 0.1);

    std::cout << "=== TEST 6 : MLP Classification (XOR) ===\n";
    for (int i = 0; i < 4; i++) {
        std::vector<double> sample = {X_xor_mlp[i*2], X_xor_mlp[i*2+1]};
        std::vector<double> pred = mlp_classif.predict(sample);
        double class_pred = (pred[0] >= 0.0) ? 1.0 : -1.0;
        std::cout << "  x={" << sample[0] << "," << sample[1] << "} "
                  << "pred_brute=" << pred[0] << "\tclasse=" << class_pred
                  << "\tattendu=" << Y_xor_mlp[i] << "\n";
    }
    std::cout << "  (L'Accuracy doit etre de 4/4)\n\n";

    std::vector<double> X_quad_mlp = {-2.0, -1.0, 0.0, 1.0, 2.0};
    std::vector<double> Y_quad_mlp = {4.0, 1.0, 0.0, 1.0, 4.0};

    MLP mlp_reg({1, 8, 1}, false);
    mlp_reg.train(X_quad_mlp, Y_quad_mlp, 30000, 0.01);

    std::cout << "=== TEST 7 : MLP Regression (y = x^2) ===\n";
    for (int i = 0; i < 5; i++) {
        std::vector<double> sample = {X_quad_mlp[i]};
        std::vector<double> pred = mlp_reg.predict(sample);
        std::cout << "  x=" << sample[0] << "\tpred=" << pred[0] << "\t\tattendu=" << Y_quad_mlp[i] << "\n";
    }
    std::cout << "  (les predictions doivent etre proches des attendus)\n\n";

    // ==========================================
    // TESTS SVM / SVR
    // ==========================================
    std::vector<std::vector<double>> X_svm = {
        { 1.0,  1.0},
        { 2.0,  2.0},
        {-1.0, -1.0},
        {-2.0, -2.0}
    };
    std::vector<double> Y_svm = {1.0, 1.0, -1.0, -1.0};

    std::vector<std::vector<double>> X_xor_2d = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    std::vector<std::vector<double>> X_xor_trans_2d;
    for (int i = 0; i < 4; i++) {
        double x0 = X_xor_2d[i][0];
        double x1 = X_xor_2d[i][1];
        X_xor_trans_2d.push_back({x0, x1, x0 * x1});
    }

    std::vector<std::vector<double>> X_reg_2d;
    for (int i = 0; i < 5; i++)
        X_reg_2d.push_back({X_reg[i]});

    std::vector<std::vector<double>> X_quad_2d;
    for (int i = 0; i < 5; i++)
        X_quad_2d.push_back({X_quad[i]});

    // TEST 6 : SVM separable
    SVM svm(2, 1.0);
    svm.train(X_svm, Y_svm, 0.01, 1000);

    std::cout << "=== TEST 6 : SVM separable ===\n";
    int correct_6 = 0;
    for (int i = 0; i < static_cast<int>(X_svm.size()); i++) {
        double pred = svm.predict(X_svm[i]);
        bool ok = (pred == Y_svm[i]);
        if (ok) correct_6++;
        std::cout << "  pred=" << pred << "  attendu=" << Y_svm[i]
                  << (ok ? "  OK" : "  ERREUR") << "\n";
    }
    std::cout << "  Accuracy : " << correct_6 << "/4 (attendu : 4/4)\n\n";

    // TEST 7 : SVM XOR (echec attendu)
    SVM svm_xor(2, 1.0);
    svm_xor.train(X_xor_2d, Y_xor, 0.01, 1000);

    std::cout << "=== TEST 7 : SVM XOR (echec attendu) ===\n";
    int correct_7 = 0;
    for (int i = 0; i < static_cast<int>(X_xor_2d.size()); i++) {
        double pred = svm_xor.predict(X_xor_2d[i]);
        bool ok = (pred == Y_xor[i]);
        if (ok) correct_7++;
        std::cout << "  pred=" << pred << "  attendu=" << Y_xor[i]
                  << (ok ? "  OK" : "  ERREUR") << "\n";
    }
    std::cout << "  Accuracy : " << correct_7 << "/4 (attendu : < 4/4)\n\n";

    // TEST 8 : SVM XOR transforme
    // FIX : C=1.0 -> C=0.001. Avec C=1.0, la regularisation (2*lr*C*w a chaque pas)
    // ecrasait les poids avant que le gradient de la hinge loss ne puisse les faire grandir.
    SVM svm_xor_t(3, 0.001);
    svm_xor_t.train(X_xor_trans_2d, Y_xor, 0.1, 5000);

    std::cout << "=== TEST 8 : SVM XOR transforme ===\n";
    int correct_8 = 0;
    for (int i = 0; i < static_cast<int>(X_xor_trans_2d.size()); i++) {
        double pred = svm_xor_t.predict(X_xor_trans_2d[i]);
        bool ok = (pred == Y_xor[i]);
        if (ok) correct_8++;
        std::cout << "  pred=" << pred << "  attendu=" << Y_xor[i]
                  << (ok ? "  OK" : "  ERREUR") << "\n";
    }
    std::cout << "  Accuracy : " << correct_8 << "/4 (attendu : 4/4)\n\n";

    // TEST 9 : SVR lineaire y = 2x + 1
    SVM svr(1, 0.00001, 0.00001, SVM::REGRESSION);  // C=0 (aucune regularisation), epsilon quasi nul
    svr.train(X_reg_2d, Y_reg, 0.00005, 500000);   // lr tres petit, beaucoup d'epochs pour compenser

    std::cout << "=== TEST 9 : SVR lineaire (y = 2x + 1) ===\n";
    for (int i = 0; i < static_cast<int>(X_reg_2d.size()); i++) {
        double pred = svr.predict(X_reg_2d[i]);
        std::cout << "  x=" << X_reg_2d[i][0]
                  << "  pred=" << pred
                  << "  attendu=" << Y_reg[i] << "\n";
    }
    std::cout << "  (predictions proches des attendus)\n\n";

    // TEST 10 : SVR y = x2 (echec attendu)
    SVM svr_quad(1, 1.0, 0.1, SVM::REGRESSION);
    svr_quad.train(X_quad_2d, Y_quad, 0.01, 1000);

    std::cout << "=== TEST 10 : SVR y=x2 (echec attendu) ===\n";
    for (int i = 0; i < static_cast<int>(X_quad_2d.size()); i++) {
        double pred = svr_quad.predict(X_quad_2d[i]);
        std::cout << "  x=" << X_quad_2d[i][0]
                  << "  pred=" << pred
                  << "  attendu=" << Y_quad[i] << "\n";
    }
    std::cout << "  (predictions mauvaises -- SVR trop simple pour x2)\n\n";

    // ==========================================
    // TESTS RBF
    // ==========================================
    std::cout << "\n==========================================\n";
    std::cout << "               TESTS RBF\n";
    std::cout << "==========================================\n\n";

    // TEST 11 : RBF Classification (XOR)
    // 1 centre par point (4 centres / 4 samples) => interpolation quasi-exacte
    std::vector<double> X_xor_rbf = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    };
    std::vector<double> Y_xor_rbf = {-1.0, 1.0, 1.0, -1.0};

    RBF rbf_classif(2, 4, 1, 0.0, true);
    rbf_classif.train(X_xor_rbf, Y_xor_rbf, 4);

    std::cout << "=== TEST 11 : RBF Classification (XOR) ===\n";
    int correct_11 = 0;
    for (int i = 0; i < 4; i++) {
        std::vector<double> sample = {X_xor_rbf[i*2], X_xor_rbf[i*2+1]};
        std::vector<double> pred = rbf_classif.predict(sample);
        bool ok = (pred[0] == Y_xor_rbf[i]);
        if (ok) correct_11++;
        std::cout << "  x={" << sample[0] << "," << sample[1] << "} "
                  << "pred=" << pred[0] << "  attendu=" << Y_xor_rbf[i]
                  << (ok ? "  OK" : "  ERREUR") << "\n";
    }
    std::cout << "  Accuracy : " << correct_11 << "/4 (attendu : 4/4)\n\n";

    // TEST 12 : RBF Regression (y = x^2)
    std::vector<double> X_quad_rbf = {-2.0, -1.0, 0.0, 1.0, 2.0};
    std::vector<double> Y_quad_rbf = {4.0, 1.0, 0.0, 1.0, 4.0};

    RBF rbf_reg(1, 5, 1, 0.0, false);
    rbf_reg.train(X_quad_rbf, Y_quad_rbf, 5);

    std::cout << "=== TEST 12 : RBF Regression (y = x^2) ===\n";
    for (int i = 0; i < 5; i++) {
        std::vector<double> sample = {X_quad_rbf[i]};
        std::vector<double> pred = rbf_reg.predict(sample);
        std::cout << "  x=" << sample[0]
                  << "  pred=" << pred[0]
                  << "  attendu=" << Y_quad_rbf[i] << "\n";
    }
    std::cout << "  (les predictions doivent etre proches des attendus)\n\n";

    // RESUME
    std::cout << "=== RESUME ===\n";
    std::cout << "  Separable          : " << correct_1 << "/4\n";
    std::cout << "  XOR sans transform : " << correct_2 << "/4  (echec attendu)\n";
    std::cout << "  XOR avec transform : " << correct_3 << "/4\n";
    std::cout << "  SVM separable      : " << correct_6 << "/4\n";
    std::cout << "  SVM XOR            : " << correct_7 << "/4  (echec attendu)\n";
    std::cout << "  SVM XOR transforme : " << correct_8 << "/4\n";
    std::cout << "  RBF XOR            : " << correct_11 << "/4\n";

    return 0;
}