#include <iostream>
#include <vector>
#include <cmath>
#include "LinearModel.hpp"
#include "MLP.hpp"

// ============================================================
//  UTILITAIRES
// ============================================================

std::vector<double> flatten(const std::vector<std::vector<double>>& X) {
    std::vector<double> flat;
    for (const auto& row : X)
        for (double v : row)
            flat.push_back(v);
    return flat;
}

int evaluate(LinearModel& model,
             const std::vector<std::vector<double>>& X,
             const std::vector<double>& Y) {
    int correct = 0;
    for (int i = 0; i < static_cast<int>(X.size()); i++) {
        double pred = model.predict(X[i]);
        bool ok = (pred == Y[i]);
        if (ok) correct++;
        std::cout << "  pred=" << pred
                  << "  attendu=" << Y[i]
                  << (ok ? " OK" : " ERREUR") << "\n";
    }
    return correct;
}

// ============================================================
//  TESTS LINEARMODEL
// ============================================================

int test_lineairement_separable() {
    std::cout << "=== TEST 1 : Lineairement separable ===\n";

    std::vector<std::vector<double>> X = {
        { 1.0,  1.0},
        { 2.0,  2.0},
        {-1.0, -1.0},
        {-2.0, -2.0}
    };
    std::vector<double> Y = {1.0, 1.0, -1.0, -1.0};

    LinearModel model(2);
    model.train(flatten(X), Y, 0.1, 50);

    int correct = evaluate(model, X, Y);
    std::cout << "Accuracy : " << correct << "/4 (attendu : 4/4)\n\n";
    return correct;
}

int test_xor_sans_transformation() {
    std::cout << "=== TEST 2 : XOR (echec attendu) ===\n";

    std::vector<std::vector<double>> X_xor = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<double> Y_xor = {-1.0, 1.0, 1.0, -1.0};

    LinearModel model_xor(2);
    model_xor.train(flatten(X_xor), Y_xor, 0.1, 100);

    int correct = evaluate(model_xor, X_xor, Y_xor);
    std::cout << "Accuracy : " << correct << "/4 (attendu : max 2/4)\n\n";
    return correct;
}

int test_xor_avec_transformation() {
    std::cout << "=== TEST 3 : XOR avec transformation non-lineaire ===\n";

    std::vector<std::vector<double>> X_xor = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<double> Y_xor = {-1.0, 1.0, 1.0, -1.0};

    std::vector<std::vector<double>> X_xor_trans;
    for (int i = 0; i < static_cast<int>(X_xor.size()); i++) {
        double x0 = X_xor[i][0];
        double x1 = X_xor[i][1];
        X_xor_trans.push_back({x0, x1, x0 * x1});
    }

    LinearModel model_trans(3);
    model_trans.train(flatten(X_xor_trans), Y_xor, 0.1, 500);

    int correct = evaluate(model_trans, X_xor_trans, Y_xor);
    std::cout << "Accuracy : " << correct << "/4 (attendu : 4/4)\n\n";
    return correct;
}

// ============================================================
//  TESTS MLP
// ============================================================

// Test 4 : XOR avec MLP — doit réussir sans transformation manuelle
int test_mlp_xor() {
    std::cout << "=== TEST 4 : MLP XOR ===\n";

    std::vector<double> X = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    };
    std::vector<double> Y = {-1.0, 1.0, 1.0, -1.0};

    MLP model({2, 3, 1});
    model.train(X, Y, 10000, 0.1, true);

    int correct = 0;
    std::vector<std::vector<double>> samples = {{0,0},{0,1},{1,0},{1,1}};
    for (int i = 0; i < 4; i++) {
        double pred = model.predict(samples[i], true)[0];
        double expected = Y[i];
        bool ok = (pred >= 0 && expected == 1.0) || (pred < 0 && expected == -1.0);
        if (ok) correct++;
        std::cout << "  pred=" << pred << "  attendu=" << expected << (ok ? " OK" : " ERREUR") << "\n";
    }
    std::cout << "Accuracy : " << correct << "/4 (attendu : 4/4)\n\n";
    return correct;
}

// Test 5 : Régression simple — le MLP doit apprendre y = x
int test_mlp_regression() {
    std::cout << "=== TEST 5 : MLP Regression ===\n";

    std::vector<double> X = {0.0, 0.25, 0.5, 0.75, 1.0};
    std::vector<double> Y = {0.0, 0.25, 0.5, 0.75, 1.0};

    MLP model({1, 5, 1});
    model.train(X, Y, 50000, 0.01, false);

    int correct = 0;
    for (int i = 0; i < 5; i++) {
        double pred = model.predict({X[i]}, false)[0];
        bool ok = std::abs(pred - Y[i]) < 0.1;
        if (ok) correct++;
        std::cout << "  pred=" << pred << "  attendu=" << Y[i] << (ok ? " OK" : " ERREUR") << "\n";
    }
    std::cout << "Accuracy : " << correct << "/5\n\n";
    return correct;
}

// Test 6 : Sauvegarde et chargement du modèle MLP
int test_mlp_save_load() {
    std::cout << "=== TEST 6 : MLP Save/Load ===\n";

    std::vector<double> X = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    };
    std::vector<double> Y = {-1.0, 1.0, 1.0, -1.0};

    // Entraîner et sauvegarder
    MLP model({2, 3, 1});
    model.train(X, Y, 10000, 0.1, true);
    model.save("mlp_test.txt");

    // Charger dans un nouveau modèle et comparer
    MLP model_loaded({2, 3, 1});
    model_loaded.load("mlp_test.txt");

    int correct = 0;
    std::vector<std::vector<double>> samples = {{0,0},{0,1},{1,0},{1,1}};
    for (int i = 0; i < 4; i++) {
        double pred_orig   = model.predict(samples[i], true)[0];
        double pred_loaded = model_loaded.predict(samples[i], true)[0];
        bool ok = std::abs(pred_orig - pred_loaded) < 1e-6;
        if (ok) correct++;
        std::cout << "  original=" << pred_orig << "  chargé=" << pred_loaded << (ok ? " OK" : " ERREUR") << "\n";
    }
    std::cout << "Accuracy : " << correct << "/4 (attendu : 4/4)\n\n";
    return correct;
}

int main() {
    int r1 = test_lineairement_separable();
    int r2 = test_xor_sans_transformation();
    int r3 = test_xor_avec_transformation();
    int r4 = test_mlp_xor();
    int r5 = test_mlp_regression();
    int r6 = test_mlp_save_load();

    std::cout << "=== RESUME ===\n";
    std::cout << "  Separable          : " << r1 << "/4\n";
    std::cout << "  XOR sans transform : " << r2 << "/4\n";
    std::cout << "  XOR avec transform : " << r3 << "/4\n";
    std::cout << "  MLP XOR            : " << r4 << "/4\n";
    std::cout << "  MLP Regression     : " << r5 << "/5\n";
    std::cout << "  MLP Save/Load      : " << r6 << "/4\n";

    return 0;
}