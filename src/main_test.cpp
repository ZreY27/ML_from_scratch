#include <iostream>
#include "LinearModel.h"

int main() {
    //Test 1 données linéairement séparables, La frontière de décision est x0 + x1 = 0, Tout ce qui est au dessus → classe +1, Tout ce qui est en dessous → classe -1
    std::vector<std::vector<double>> X = {
        { 1.0,  1.0},  // somme =  2.0 > 0 → classe +1
        { 2.0,  2.0},  // somme =  4.0 > 0 → classe +1
        {-1.0, -1.0},  // somme = -2.0 < 0 → classe -1
        {-2.0, -2.0}   // somme = -4.0 < 0 → classe -1
    };
    std::vector<double> Y = {1.0,1.0,-1.0,-1.0};
    LinearModel model(2);// 2 = nombre de features d'entrée (x0 et x1) , crée weights = {random, random}, bias = 0.0
    model.train(X,Y,0.1,50);
    std::cout << "=== TEST 1 : Lineairement separable ===\n";
    int correct_1 = 0;
    for (int i = 0; static_cast<int>(X.size()); i++) {
        double pred = model.predict(X[i]);
        bool ok = (pred == Y[i]);
        if (ok) correct_1++;
        std::cout << "  pred=" << pred
                  << "  attendu=" << Y[i]
                  << (ok ? " OK" : "ERREUR") << "\n";

    }
    std::cout << "Accuracy : " << correct_1 << "/4"
                  << "(attendu : 4/4)\n\n";
    // Test 2 : XOR,  n'est PAS linéairement séparable
    // Le modèle DOIT échouer ici — c'est voulu
    // → prouve qu'on comprend la limite du modèle linéaire
    // → justifie l'ajout de la transformation non-linéaire après
    std::vector<std::vector<double>> X_xor = {
        {0.0, 0.0},  // XOR(0,0) = 0 → classe -1
        {0.0, 1.0},  // XOR(0,1) = 1 → classe +1
        {1.0, 0.0},  // XOR(1,0) = 1 → classe +1
        {1.0, 1.0}   // XOR(1,1) = 0 → classe -1
    };
    std::vector<double> Y_xor = {-1.0, 1.0, 1.0, -1.0};
    LinearModel model_xor(2);
    model_xor.train(X_xor,Y_xor,0.1,100);
    std::cout << "=== TEST 2 : XOR (echec attendu) ===\n";
    int correct_2 = 0;
    for (int i = 0; static_cast<int>(X_xor.size()); i++) {
        double pred = model.predict(X_xor[i]);
        bool ok = (pred == Y_xor[i]);
        if (ok) correct_2++;
        std::cout << "  pred=" << pred
                  << "  attendu=" << Y_xor[i]
                  << (ok ? " OK" : "ERREUR") << "\n";
    }
    std::cout << "Accuracy : " << correct_2 << "/4"
    << "(attendu : 4/4)\n\n";

    //Test3 : XOR avec transformation non-linéaire
    // On ajoute le feature x0*x1 aux features de base
    // [x0, x1] → [x0, x1, x0*x1]
    // Avec ce feature supplémentaire le problème devient, linéairement séparable → le modèle DOIT réussir
    std::vector<std::vector<double>> X_xor_trans;
    for (int i = 0; static_cast<int>(X_xor.size()); i++) {
        double x0 = X_xor[i][0];
        double x1 = X_xor[i][1];
        X_xor_trans.push_back({x0,x1,x0 * x1});
    }// on ajoute le terme croisé x0*x1 comme 3ème feature
    LinearModel model_trans(3);
    model_trans.train(X_xor_trans, Y_xor, 0.1, 500); // plus d'epochs car le problème est plus complexe
    std::cout << "=== TEST 3 : XOR avec transformation ===\n";
    int correct_3 = 0;
    for (int i = 0; static_cast<int>(X_xor_trans.size()); i++) {
        double pred = model_trans.predict(X_xor_trans[i]);
        bool ok = (pred == Y_xor[i]);
        if (ok) correct_3++;
        std::cout << "  pred=" << pred
                  << "  attendu=" << Y_xor[i]
                  << (ok ? " OK" : "ERREUR") << "\n";
    }
    std::cout << "Accuracy : " << correct_3 << "/4"
              << "(attendu : 4/4)\n\n";


    std::cout << "=== RESUME ===\n";
    std::cout << "  Separable          : " << correct_1 << "/4\n";
    std::cout << "  XOR sans transform : " << correct_2 << "/4\n";
    std::cout << "  XOR avec transform : " << correct_3 << "/4\n";

    return 0;
}