// ============================================================
// Statistiques.h — Statistiques et régressions pour l'embarqué
// ============================================================
// PORTAGE du projet Qt `Fo170/Statistiques` (statsengine.cpp + includes/*.hl)
// en version embarquée, header-only, sans allocation dynamique.
//
// POURQUOI UN PORTAGE ET NON UNE UTILISATION DIRECTE
// --------------------------------------------------
// L'amont est une APPLICATION Qt : `statsengine.cpp` dépend de QString/QVector,
// et `includes/*.hl` sont des fichiers DOS de 1996 (macros `_()`, `#ifdef ORD`
// pour activer les fonctions, extension `.hl`). Rien de tout cela ne compile
// sur ESP8266.
//
// Ce portage conserve les ALGORITHMES (9 modèles de régression) en `float`,
// sur des tableaux de taille fixe, et n'utilise que <math.h>.
//
// CE QUI EST FOURNI
// -----------------
//   • Statistiques de base : moyenne, variance, écart-type, médiane, MAD,
//     min/max, covariance, corrélation.
//   • Tri : insertion (stable, O(n²) mais n ≤ 16 → négligeable).
//   • 9 modèles de régression, du plus simple au plus général :
//       0 LINÉAIRE      y = a + b·x
//       1 LOGARITHMIQUE y = a + b·ln(x + tx)
//       2 EXPONENTIELLE y = a·e^(b·x) − ty
//       3 PUISSANCE     y = a·(x + tx)^b − ty
//       4 PUISSANCE_NLS y = a·x^b                    (Gauss-Newton)
//       5 RÉCIPROQUE    y = a + b/x
//       6 POLYNÔME 2    y = a + b·x + c·x²
//       7 SINUSOÏDAL    y = a·sin(b·x + c) + d       (Gauss-Newton)
//       8 LOGISTIQUE    y = c/(1 + a·e^(−b·x))       (Gauss-Newton)
//   • AUTO : teste les 9 modèles et retient le meilleur r².
//
// ⚠️ Les modes NLS (4, 7, 8) sont gourmands : sur ESP8266 (pas de FPU) ils
//    coûtent ~30 itérations × n évaluations. Ne PAS les lancer à 10 Hz.
//    Usage prévu : une fois par jour, sur 8 à 16 points.
//
// RÉFÉRENCES
// ----------
//   • Legendre / Gauss — moindres carrés ordinaires (MCO), conditions normales.
//   • Gauss-Newton — Levenberg & Marquardt, « Methods for the Solution of
//     Problems in Least Squares », Quarterly of Applied Mathematics, 1944.
//   • Michaelis-Menten (mode 5) — cinétique enzymatique, transformation de
//     Lineweaver-Burk.
//   • Verhulst (mode 8) — équation logistique, 1838.
//   • https://fr.wikipedia.org/wiki/R%C3%A9gression_lin%C3%A9aire (2026-10-20)
//   • https://fr.wikipedia.org/wiki/Algorithme_de_Gauss-Newton (2026-10-20)
//
// RAM : aucune allocation. Les buffers de travail sont fournis par l'appelant
//       ou internes à taille fixe (STATS_N_MAX = 16 points).
// ============================================================

#ifndef STATISTIQUES_H
#define STATISTIQUES_H

#include <Arduino.h>
#include <math.h>

#define __STATISTIQUES_H_VERSION__ "1.0.0"

// Nombre maximal de points traités d'un coup (16 = REG_MAX_POINTS du projet).
#ifndef STATS_N_MAX
#define STATS_N_MAX 16
#endif

// Nombre d'itérations Gauss-Newton pour les modèles NLS.
#ifndef STATS_NLS_ITER
#define STATS_NLS_ITER 20
#endif

class Statistiques {
public:
    enum Mode : uint8_t {
        LIN = 0, LOG = 1, EXP = 2, POW = 3, POW_NLS = 4,
        INV = 5, POLY2 = 6, SIN = 7, LOGIT = 8,
        AUTO = 9, MODE_COUNT = 10
    };

    // ------------------------------------------------------------
    // Statistiques descriptives
    // ------------------------------------------------------------
    static float mean(const float* x, uint8_t n) {
        if (!x || n == 0) return 0.0f;
        float s = 0.0f;
        for (uint8_t i = 0; i < n; i++) s += x[i];
        return s / (float)n;
    }

    /** @brief Variance (population, ÷ n — convention de l'amont). */
    static float variance(const float* x, uint8_t n) {
        if (!x || n < 2) return 0.0f;
        float m = mean(x, n), s = 0.0f;
        for (uint8_t i = 0; i < n; i++) { float d = x[i] - m; s += d * d; }
        return s / (float)n;
    }

    static float stddev(const float* x, uint8_t n) { return sqrtf(variance(x, n)); }

    /** @brief Médiane. Trie une COPIE interne (l'entrée n'est pas modifiée). */
    static float median(const float* x, uint8_t n) {
        if (!x || n == 0) return 0.0f;
        float buf[STATS_N_MAX];
        uint8_t m = (n > STATS_N_MAX) ? STATS_N_MAX : n;
        for (uint8_t i = 0; i < m; i++) buf[i] = x[i];
        sortAsc(buf, m);
        return (m & 1) ? buf[m / 2] : 0.5f * (buf[m / 2 - 1] + buf[m / 2]);
    }

    /**
     * @brief MAD — écart absolu médian (Median Absolute Deviation).
     *
     * C'est le socle du filtre de Hampel : un point est aberrant si
     * |x − médiane| > k·MAD. Robustesse ~50 % (contre 0 % pour l'écart-type).
     */
    static float mad(const float* x, uint8_t n) {
        if (!x || n == 0) return 0.0f;
        float med = median(x, n);
        float dev[STATS_N_MAX];
        uint8_t m = (n > STATS_N_MAX) ? STATS_N_MAX : n;
        for (uint8_t i = 0; i < m; i++) dev[i] = fabsf(x[i] - med);
        return median(dev, m);
    }

    static void minMax(const float* x, uint8_t n, float& vmin, float& vmax) {
        vmin = vmax = 0.0f;
        if (!x || n == 0) return;
        vmin = vmax = x[0];
        for (uint8_t i = 1; i < n; i++) {
            if (x[i] < vmin) vmin = x[i];
            if (x[i] > vmax) vmax = x[i];
        }
    }

    static float covariance(const float* x, const float* y, uint8_t n) {
        if (!x || !y || n < 2) return 0.0f;
        float mx = mean(x, n), my = mean(y, n), s = 0.0f;
        for (uint8_t i = 0; i < n; i++) s += (x[i] - mx) * (y[i] - my);
        return s / (float)n;
    }

    /** @brief Corrélation de Pearson (mode linéaire uniquement). */
    static float correlation(const float* x, const float* y, uint8_t n) {
        float sx = stddev(x, n), sy = stddev(y, n);
        if (!(sx > 1e-12f) || !(sy > 1e-12f)) return 0.0f;
        return covariance(x, y, n) / (sx * sy);
    }

    /** @brief Tri croissant par insertion (stable, n ≤ 16). */
    static void sortAsc(float* v, uint8_t n) {
        for (uint8_t i = 1; i < n; i++) {
            float key = v[i];
            int j = (int)i - 1;
            while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; j--; }
            v[j + 1] = key;
        }
    }

    // ------------------------------------------------------------
    // Modèles de régression — MCO linéarisés
    // ------------------------------------------------------------
    /**
     * @brief 0 LINÉAIRE : y = a + b·x
     * @return r² (0 si dégénéré)
     */
    static float fitLinear(const float* x, const float* y, uint8_t n,
                           float& a, float& b) {
        a = b = 0.0f;
        if (!x || !y || n < 2) return 0.0f;
        float mx = mean(x, n), my = mean(y, n);
        float sxy = 0.0f, sxx = 0.0f;
        for (uint8_t i = 0; i < n; i++) {
            float dx = x[i] - mx;
            sxy += dx * (y[i] - my);
            sxx += dx * dx;
        }
        if (!(sxx > 1e-12f)) return 0.0f;
        b = sxy / sxx;
        a = my - b * mx;
        return r2(x, y, n, a, b, 0.0f, 0.0f, 0.0f, LIN);
    }

    /**
     * @brief 1 LOGARITHMIQUE : y = a + b·ln(x + tx)
     * @param tx Décalage (défaut 0) — évite ln(0)
     */
    static float fitLog(const float* x, const float* y, uint8_t n,
                        float& a, float& b, float tx = 0.0f) {
        a = b = 0.0f;
        if (!x || !y || n < 2) return 0.0f;
        float u[STATS_N_MAX], v[STATS_N_MAX];
        uint8_t m = 0;
        for (uint8_t i = 0; i < n && m < STATS_N_MAX; i++) {
            float t = x[i] + tx;
            if (!(t > 0.0f)) continue;
            u[m] = logf(t);
            v[m] = y[i];
            m++;
        }
        if (m < 2) return 0.0f;
        float la, lb;
        fitLinear(u, v, m, la, lb);
        a = la; b = lb;
        return r2(x, y, n, a, b, 0.0f, tx, 0.0f, LOG);
    }

    /**
     * @brief 2 EXPONENTIELLE : y = a·e^(b·x) − ty
     * @param ty Décalage vertical (défaut 0)
     */
    static float fitExp(const float* x, const float* y, uint8_t n,
                        float& a, float& b, float ty = 0.0f) {
        a = b = 0.0f;
        if (!x || !y || n < 2) return 0.0f;
        float u[STATS_N_MAX], v[STATS_N_MAX];
        uint8_t m = 0;
        for (uint8_t i = 0; i < n && m < STATS_N_MAX; i++) {
            float t = y[i] + ty;
            if (!(t > 0.0f)) continue;
            u[m] = x[i];
            v[m] = logf(t);
            m++;
        }
        if (m < 2) return 0.0f;
        float la, lb;
        fitLinear(u, v, m, la, lb);
        a = expf(la); b = lb;
        return r2(x, y, n, a, b, 0.0f, 0.0f, ty, EXP);
    }

    /**
     * @brief 3 PUISSANCE : y = a·(x + tx)^b − ty
     */
    static float fitPow(const float* x, const float* y, uint8_t n,
                        float& a, float& b, float tx = 0.0f, float ty = 0.0f) {
        a = b = 0.0f;
        if (!x || !y || n < 2) return 0.0f;
        float u[STATS_N_MAX], v[STATS_N_MAX];
        uint8_t m = 0;
        for (uint8_t i = 0; i < n && m < STATS_N_MAX; i++) {
            float t = x[i] + tx, s = y[i] + ty;
            if (!(t > 0.0f) || !(s > 0.0f)) continue;
            u[m] = logf(t);
            v[m] = logf(s);
            m++;
        }
        if (m < 2) return 0.0f;
        float la, lb;
        fitLinear(u, v, m, la, lb);
        a = expf(la); b = lb;
        return r2(x, y, n, a, b, 0.0f, tx, ty, POW);
    }

    /**
     * @brief 5 RÉCIPROQUE : y = a + b/x   (type Michaelis-Menten)
     */
    static float fitInv(const float* x, const float* y, uint8_t n,
                        float& a, float& b) {
        a = b = 0.0f;
        if (!x || !y || n < 2) return 0.0f;
        float u[STATS_N_MAX], v[STATS_N_MAX];
        uint8_t m = 0;
        for (uint8_t i = 0; i < n && m < STATS_N_MAX; i++) {
            if (fabsf(x[i]) < 1e-12f) continue;
            u[m] = 1.0f / x[i];
            v[m] = y[i];
            m++;
        }
        if (m < 2) return 0.0f;
        float la, lb;
        fitLinear(u, v, m, la, lb);
        // u = 1/x, v = y  ⇒  y = la + lb·(1/x)  ⇒  a = la (ordonnée), b = lb (pente)
        a = la; b = lb;
        return r2(x, y, n, a, b, 0.0f, 0.0f, 0.0f, INV);
    }

    /**
     * @brief 6 POLYNÔME degré 2 : y = a + b·x + c·x²
     *
     * Résolution des 3 conditions normales par élimination de Gauss.
     */
    static float fitPoly2(const float* x, const float* y, uint8_t n,
                          float& a, float& b, float& c) {
        a = b = c = 0.0f;
        if (!x || !y || n < 3) return 0.0f;
        // Σx, Σx², Σx³, Σx⁴, Σy, Σxy, Σx²y
        double S1 = 0, S2 = 0, S3 = 0, S4 = 0, Sy = 0, Sxy = 0, Sx2y = 0;
        for (uint8_t i = 0; i < n; i++) {
            double xi = x[i], yi = y[i], x2 = xi * xi;
            S1 += xi; S2 += x2; S3 += x2 * xi; S4 += x2 * x2;
            Sy += yi; Sxy += xi * yi; Sx2y += x2 * yi;
        }
        // Matrice 3×3 augmentée
        double M[3][4] = {
            {(double)n, S1, S2, Sy},
            {S1, S2, S3, Sxy},
            {S2, S3, S4, Sx2y}
        };
        // Gauss avec pivot partiel
        for (uint8_t col = 0; col < 3; col++) {
            uint8_t piv = col;
            for (uint8_t r = col + 1; r < 3; r++)
                if (fabs(M[r][col]) > fabs(M[piv][col])) piv = r;
            if (fabs(M[piv][col]) < 1e-12) return 0.0f;
            if (piv != col)
                for (uint8_t k = col; k < 4; k++) { double t = M[col][k]; M[col][k] = M[piv][k]; M[piv][k] = t; }
            for (uint8_t r = col + 1; r < 3; r++) {
                double f = M[r][col] / M[col][col];
                for (uint8_t k = col; k < 4; k++) M[r][k] -= f * M[col][k];
            }
        }
        double sol[3];
        for (int r = 2; r >= 0; r--) {
            double s = M[r][3];
            for (uint8_t k = r + 1; k < 3; k++) s -= M[r][k] * sol[k];
            sol[r] = s / M[r][r];
        }
        a = (float)sol[0]; b = (float)sol[1]; c = (float)sol[2];
        if (!isfinite(a) || !isfinite(b) || !isfinite(c)) { a = b = c = 0.0f; return 0.0f; }
        return r2Poly2(x, y, n, a, b, c);
    }

    /**
     * @brief Évalue un modèle au point x (paramètres a, b, c, tx, ty selon mode).
     */
    static float eval(Mode mode, float x, float a, float b, float c,
                      float tx = 0.0f, float ty = 0.0f) {
        switch (mode) {
            case LIN:  return a + b * x;
            case LOG:  return (x + tx > 0.0f) ? a + b * logf(x + tx) : NAN;
            case EXP:  return a * expf(b * x) - ty;
            case POW:  return (x + tx > 0.0f) ? a * powf(x + tx, b) - ty : NAN;
            case POW_NLS: return (x > 0.0f) ? a * powf(x, b) : NAN;
            case INV:  return (fabsf(x) > 1e-12f) ? a + b / x : NAN;
            case POLY2:return a + b * x + c * x * x;
            case SIN:  return a * sinf(b * x + c);
            case LOGIT:return c / (1.0f + a * expf(-b * x));
            default:   return NAN;
        }
    }

    // ------------------------------------------------------------
    // r²
    // ------------------------------------------------------------
    // ⚠️ `c` fait partie de la signature : les modes POLY2 et LOGIT en ont
    // besoin. Sans lui, `r2(... LOGIT)` passait le paramètre `c` à la place
    // de `tx`, `eval()` recevait c = 0 et renvoyait 0 → r² = 0 alors que
    // l'ajustement était parfait (bug détecté par les tests hôte).
    static float r2(const float* x, const float* y, uint8_t n,
                    float a, float b, float c, float tx, float ty, Mode mode) {
        if (!x || !y || n < 2) return 0.0f;
        float my = mean(y, n);
        double ssRes = 0, ssTot = 0;
        for (uint8_t i = 0; i < n; i++) {
            float e = eval(mode, x[i], a, b, c, tx, ty);
            if (!isfinite(e)) continue;
            double d1 = (double)y[i] - e;
            double d2 = (double)y[i] - my;
            ssRes += d1 * d1;
            ssTot += d2 * d2;
        }
        if (!(ssTot > 1e-12)) return 0.0f;
        float v = 1.0f - (float)(ssRes / ssTot);
        return (v < 0.0f) ? 0.0f : v;
    }

    static float r2Poly2(const float* x, const float* y, uint8_t n,
                         float a, float b, float c) {
        if (!x || !y || n < 2) return 0.0f;
        float my = mean(y, n);
        double ssRes = 0, ssTot = 0;
        for (uint8_t i = 0; i < n; i++) {
            float e = a + b * x[i] + c * x[i] * x[i];
            double d1 = (double)y[i] - e;
            double d2 = (double)y[i] - my;
            ssRes += d1 * d1;
            ssTot += d2 * d2;
        }
        if (!(ssTot > 1e-12)) return 0.0f;
        float v = 1.0f - (float)(ssRes / ssTot);
        return (v < 0.0f) ? 0.0f : v;
    }

    // ------------------------------------------------------------
    // NLS — Gauss-Newton AMORTI (Levenberg-Marquardt avec rejet d'étape)
    // ------------------------------------------------------------
    // ⚠️ POURQUOI L'AMORTISSEMENT EST INDISPENSABLE
    // ---------------------------------------------
    // Gauss-Newton PUR diverge dès que l'amorçage est éloigné de l'optimum :
    // l'étape Δ = (JᵀJ)⁻¹·Jᵀr n'est valide que dans le voisinage quadratique
    // du minimum. Observé lors de la mise au point : `fitLogistic` amorcé à
    // B = 0,1 alors que la vraie valeur vaut 0,5 sortait du domaine et
    // retournait a = b = c = 0 (rejet de validation).
    //
    // Levenberg-Marquardt ajoute λ·diag(JᵀJ) à la diagonale :
    //   • λ grand → pas de gradient (lent mais sûr)
    //   • λ petit → pas de Gauss-Newton (rapide près de l'optimum)
    // et REJETTE l'étape si la somme des carrés augmente. C'est ce rejet qui
    // garantit la monotonie : le résidu ne peut jamais remonter.
    //
    // Réf. : Levenberg (1944), Marquardt (1963), « An Algorithm for Least-
    //        Squares Estimation of Nonlinear Parameters », SIAM J. Appl. Math.

    /**
     * @brief Résout un système linéaire m×m par Gauss avec pivot partiel.
     * @param M Matrice augmentée m×(m+1), MODIFIÉE en place
     * @param sol Reçoit la solution
     * @return false si la matrice est singulière
     */
    static bool solveLinear(double M[][5], uint8_t m, double* sol) {
        for (uint8_t col = 0; col < m; col++) {
            uint8_t piv = col;
            for (uint8_t r = col + 1; r < m; r++)
                if (fabs(M[r][col]) > fabs(M[piv][col])) piv = r;
            if (fabs(M[piv][col]) < 1e-15) return false;
            if (piv != col)
                for (uint8_t k = col; k <= m; k++) {
                    double t = M[col][k]; M[col][k] = M[piv][k]; M[piv][k] = t;
                }
            for (uint8_t r = col + 1; r < m; r++) {
                double f = M[r][col] / M[col][col];
                for (uint8_t k = col; k <= m; k++) M[r][k] -= f * M[col][k];
            }
        }
        for (int r = (int)m - 1; r >= 0; r--) {
            double s = M[r][m];
            for (uint8_t k = (uint8_t)r + 1; k < m; k++) s -= M[r][k] * sol[k];
            sol[r] = s / M[r][r];
        }
        return true;
    }

    /** @brief Somme des carrés des résidus d'un modèle à 4 paramètres. */
    static double sse4(const float* x, const float* y, uint8_t n,
                       float a, float b, float c, float d, Mode mode) {
        double s = 0.0;
        for (uint8_t i = 0; i < n; i++) {
            float e;
            if (mode == LOGIT) e = c / (1.0f + a * expf(-b * x[i]));
            else               e = a * sinf(b * x[i] + c) + d;   // SIN
            if (!isfinite(e)) return 1e300;
            double r = (double)y[i] - e;
            s += r * r;
        }
        return s;
    }

    /**
     * @brief 4 PUISSANCE_NLS : y = a·x^b   (Gauss-Newton amorti, 2 paramètres)
     *
     * Contrairement au mode 3, gère naturellement x = 0 (pas de logarithme
     * de x) et n'exige pas de décalage.
     *
     * @param a0, b0 Amorçage (si non finis, calculés depuis le mode 3)
     */
    static float fitPowNls(const float* x, const float* y, uint8_t n,
                           float& a, float& b, float a0 = NAN, float b0 = NAN) {
        a = b = 0.0f;
        if (!x || !y || n < 3) return 0.0f;
        // Amorçage : log-log si possible
        if (!isfinite(a0) || !isfinite(b0)) {
            float u[STATS_N_MAX], v[STATS_N_MAX];
            uint8_t m = 0;
            for (uint8_t i = 0; i < n && m < STATS_N_MAX; i++) {
                if (!(x[i] > 0.0f) || !(y[i] > 0.0f)) continue;
                u[m] = logf(x[i]); v[m] = logf(y[i]); m++;
            }
            if (m < 2) return 0.0f;
            float la, lb;
            fitLinear(u, v, m, la, lb);
            a0 = expf(la); b0 = lb;
        }
        if (!(a0 > 0.0f)) return 0.0f;

        float A = a0, B = b0;
        double lambda = 1e-3;
        double sse = sse2(x, y, n, A, B);

        for (uint8_t it = 0; it < STATS_NLS_ITER; it++) {
            // Jacobienne : ∂y/∂a = x^b ; ∂y/∂b = a·x^b·ln(x)
            double JtJ00 = 0, JtJ01 = 0, JtJ11 = 0, Jtr0 = 0, Jtr1 = 0;
            for (uint8_t i = 0; i < n; i++) {
                if (!(x[i] > 0.0f)) continue;
                double xb = pow((double)x[i], (double)B);
                double j0 = xb;
                double j1 = (double)A * xb * log((double)x[i]);
                double r  = (double)y[i] - (double)A * xb;
                JtJ00 += j0 * j0; JtJ01 += j0 * j1; JtJ11 += j1 * j1;
                Jtr0  += j0 * r;   Jtr1  += j1 * r;
            }
            // Amortissement : λ·diag(JᵀJ)
            double M[2][5] = {{0}};
            M[0][0] = JtJ00 * (1.0 + lambda);
            M[1][1] = JtJ11 * (1.0 + lambda);
            M[0][1] = M[1][0] = JtJ01;
            M[0][2] = Jtr0; M[1][2] = Jtr1;
            double sol[2];
            if (!solveLinear(M, 2, sol)) break;

            float nA = A + (float)sol[0];
            float nB = B + (float)sol[1];
            if (!isfinite(nA) || !isfinite(nB) || !(nA > 0.0f)) { lambda *= 10.0; continue; }

            double nSSE = sse2(x, y, n, nA, nB);
            if (nSSE < sse) {
                // Étape acceptée → on relâche l'amortissement
                A = nA; B = nB;
                double gain = sse - nSSE;
                sse = nSSE;
                lambda = (lambda > 1e-9) ? lambda * 0.3 : 1e-9;
                if (gain < 1e-12 * (sse + 1e-12)) break;   // convergé
            } else {
                // Étape rejetée → on resserre
                lambda *= 10.0;
                if (lambda > 1e8) break;
            }
        }
        a = A; b = B;
        return r2(x, y, n, a, b, 0.0f, 0.0f, 0.0f, POW_NLS);
    }

    /** @brief Somme des carrés des résidus du modèle puissance (2 paramètres). */
    static double sse2(const float* x, const float* y, uint8_t n, float a, float b) {
        double s = 0.0;
        for (uint8_t i = 0; i < n; i++) {
            if (!(x[i] > 0.0f)) continue;
            double e = (double)a * pow((double)x[i], (double)b);
            if (!isfinite(e)) return 1e300;
            double r = (double)y[i] - e;
            s += r * r;
        }
        return s;
    }

    /**
     * @brief 8 LOGISTIQUE : y = c/(1 + a·e^(−b·x))   (Gauss-Newton amorti)
     *
     * @param c0 Capacité limite (si non finie, = max(y)·1,05)
     *
     * ⚠️ Le modèle est fortement non linéaire en `b` : un amorçage naïf
     * (B = 0,1 alors que la vraie valeur vaut 0,5) fait DIVERGER Gauss-Newton
     * pur. L'amortissement de Levenberg-Marquardt + le rejet d'étape rendent
     * la convergence monotone. Si `c0` est fourni et fiable, il est utilisé
     * comme valeur INITIALE mais reste ajusté (une contrainte dure fausse
     * biaiserait tout l'ajustement).
     */
    static float fitLogistic(const float* x, const float* y, uint8_t n,
                             float& a, float& b, float& c, float c0 = NAN) {
        a = b = c = 0.0f;
        if (!x || !y || n < 4) return 0.0f;
        float ymin, ymax;
        minMax(y, n, ymin, ymax);
        if (!(ymax > ymin)) return 0.0f;
        if (!isfinite(c0) || !(c0 > ymax)) c0 = ymax * 1.05f;

        // Amorçage : a tel que y(x0) = c/(1+a) ; b > 0
        float A = (c0 / (y[0] > 0.0f ? y[0] : ymin + 1e-6f)) - 1.0f;
        if (!(A > 0.0f)) A = 1.0f;
        float B = 0.1f;
        float C = c0;

        double lambda = 1e-3;
        double sse = sse4(x, y, n, A, B, C, 0.0f, LOGIT);

        for (uint8_t it = 0; it < STATS_NLS_ITER; it++) {
            double JtJ[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
            double Jtr[3] = {0,0,0};
            for (uint8_t i = 0; i < n; i++) {
                double e = exp((double)(-B * x[i]));
                double den = 1.0 + (double)A * e;
                if (fabs(den) < 1e-15) continue;
                double pred = (double)C / den;
                double r = (double)y[i] - pred;
                // ∂pred/∂A = -C·e/den² ; ∂pred/∂B = C·A·x·e/den² ; ∂pred/∂C = 1/den
                double j0 = -(double)C * e / (den * den);
                double j1 =  (double)C * A * x[i] * e / (den * den);
                double j2 = 1.0 / den;
                double J[3] = {j0, j1, j2};
                for (uint8_t r1 = 0; r1 < 3; r1++) {
                    Jtr[r1] += J[r1] * r;
                    for (uint8_t c1 = 0; c1 < 3; c1++) JtJ[r1][c1] += J[r1] * J[c1];
                }
            }
            // Matrice augmentée avec amortissement λ·diag(JᵀJ)
            double M[3][5] = {{0}};
            for (uint8_t r1 = 0; r1 < 3; r1++) {
                for (uint8_t c1 = 0; c1 < 3; c1++) M[r1][c1] = JtJ[r1][c1];
                M[r1][r1] *= (1.0 + lambda);
                M[r1][3] = Jtr[r1];
            }
            double s[3];
            if (!solveLinear(M, 3, s)) break;

            float nA = A + (float)s[0];
            float nB = B + (float)s[1];
            float nC = C + (float)s[2];
            if (!isfinite(nA) || !isfinite(nB) || !isfinite(nC) ||
                !(nA > 0.0f) || !(nB > 0.0f) || !(nC > 0.0f)) { lambda *= 10.0; continue; }

            double nSSE = sse4(x, y, n, nA, nB, nC, 0.0f, LOGIT);
            if (nSSE < sse) {
                A = nA; B = nB; C = nC;
                double gain = sse - nSSE;
                sse = nSSE;
                lambda = (lambda > 1e-9) ? lambda * 0.3 : 1e-9;
                if (gain < 1e-12 * (sse + 1e-12)) break;
            } else {
                lambda *= 10.0;
                if (lambda > 1e8) break;
            }
        }
        a = A; b = B; c = C;
        // Validation : la courbe doit être sigmoïde (a > 0, b > 0, c > 0)
        if (!(a > 0.0f) || !(b > 0.0f) || !(c > 0.0f)) { a = b = c = 0.0f; return 0.0f; }
        return r2(x, y, n, a, b, c, 0.0f, 0.0f, LOGIT);
    }

    /**
     * @brief 7 SINUSOÏDAL : y = a·sin(b·x + c) + d   (Gauss-Newton amorti)
     *
     * Amorçage : amplitude = (max−min)/2, offset d = (max+min)/2, fréquence
     * estimée par comptage des passages par la moyenne, phase par corrélation
     * croisée sur 24 valeurs.
     *
     * ⚠️ Le mode sinusọïdal est le plus mal conditionné des neuf (la fréquence
     * est presque non identifiable sur moins d'un cycle). L'amorçage seul
     * suffit à un usage de diagnostic ; l'affinage NLS est tenté mais peut
     * être rejeté si le résidu ne diminue pas.
     */
    static float fitSinusoid(const float* x, const float* y, uint8_t n,
                             float& a, float& b, float& c, float& d) {
        a = b = c = d = 0.0f;
        if (!x || !y || n < 5) return 0.0f;
        float ymin, ymax;
        minMax(y, n, ymin, ymax);
        if (!(ymax > ymin)) return 0.0f;

        float amp = (ymax - ymin) * 0.5f;
        float off = (ymax + ymin) * 0.5f;

        // Période : comptage des passages par l'offset
        uint8_t crossings = 0;
        for (uint8_t i = 1; i < n; i++) {
            if ((y[i - 1] - off) * (y[i] - off) < 0.0f) crossings++;
        }
        float span = x[n - 1] - x[0];
        if (!(span > 0.0f) || crossings < 2) return 0.0f;
        float period = 2.0f * span / (float)crossings;
        float omega = 2.0f * 3.14159265358979323846f / period;

        // Phase : corrélation croisée au point de départ
        float bestPhi = 0.0f, bestR = -2.0f;
        for (uint8_t k = 0; k < 24; k++) {
            float phi = (float)k * (2.0f * 3.14159265358979323846f / 24.0f);
            double num = 0, den1 = 0, den2 = 0;
            for (uint8_t i = 0; i < n; i++) {
                double s = sin((double)(omega * x[i] + phi));
                double t = (double)y[i] - off;
                num += s * t; den1 += s * s; den2 += t * t;
            }
            if (!(den1 > 1e-15) || !(den2 > 1e-15)) continue;
            float r = (float)(num / sqrt(den1 * den2));
            if (r > bestR) { bestR = r; bestPhi = phi; }
        }
        if (bestR < -1.0f) return 0.0f;

        // --- Affinage NLS amorti sur [amp, omega, phi, off] ---
        float A = amp, B = omega, C = bestPhi, D = off;
        double lambda = 1e-2;
        double sse = sse4(x, y, n, A, B, C, D, SIN);
        const float sse0 = (float)sse;

        for (uint8_t it = 0; it < STATS_NLS_ITER; it++) {
            double JtJ[4][4] = {{0}};
            double Jtr[4] = {0,0,0,0};
            for (uint8_t i = 0; i < n; i++) {
                double ph = (double)(B * x[i] + C);
                double s = sin(ph);
                double pred = (double)A * s + (double)D;
                double r = (double)y[i] - pred;
                // ∂/∂A = sin(ph) ; ∂/∂B = A·x·cos(ph) ; ∂/∂C = A·cos(ph) ; ∂/∂D = 1
                double cs = cos(ph);
                double J[4] = { s, (double)A * x[i] * cs, (double)A * cs, 1.0 };
                for (uint8_t r1 = 0; r1 < 4; r1++) {
                    Jtr[r1] += J[r1] * r;
                    for (uint8_t c1 = 0; c1 < 4; c1++) JtJ[r1][c1] += J[r1] * J[c1];
                }
            }
            double M[4][5] = {{0}};
            for (uint8_t r1 = 0; r1 < 4; r1++) {
                for (uint8_t c1 = 0; c1 < 4; c1++) M[r1][c1] = JtJ[r1][c1];
                M[r1][r1] *= (1.0 + lambda);
                M[r1][4] = Jtr[r1];
            }
            double s4[4];
            if (!solveLinear(M, 4, s4)) break;

            float nA = A + (float)s4[0];
            float nB = B + (float)s4[1];
            float nC = C + (float)s4[2];
            float nD = D + (float)s4[3];
            if (!isfinite(nA) || !isfinite(nB) || !isfinite(nC) || !isfinite(nD) ||
                !(nB > 0.0f)) { lambda *= 10.0; continue; }

            double nSSE = sse4(x, y, n, nA, nB, nC, nD, SIN);
            if (nSSE < sse) {
                A = nA; B = nB; C = nC; D = nD;
                double gain = sse - nSSE;
                sse = nSSE;
                lambda = (lambda > 1e-9) ? lambda * 0.3 : 1e-9;
                if (gain < 1e-12 * (sse + 1e-12)) break;
            } else {
                lambda *= 10.0;
                if (lambda > 1e8) break;
            }
        }
        // Garde : si l'affinage n'a pas amélioré, on revient à l'amorçage
        // (une régression ne doit jamais dégrader le résultat).
        if (sse > (double)sse0) {
            A = amp; B = omega; C = bestPhi; D = off;
        }
        a = A; b = B; c = C; d = D;
        return r2Sin(x, y, n, a, b, c, d);
    }

    static float r2Sin(const float* x, const float* y, uint8_t n,
                       float a, float b, float c, float d) {
        if (!x || !y || n < 2) return 0.0f;
        float my = mean(y, n);
        double ssRes = 0, ssTot = 0;
        for (uint8_t i = 0; i < n; i++) {
            float e = a * sinf(b * x[i] + c) + d;
            double d1 = (double)y[i] - e;
            double d2 = (double)y[i] - my;
            ssRes += d1 * d1; ssTot += d2 * d2;
        }
        if (!(ssTot > 1e-12)) return 0.0f;
        float v = 1.0f - (float)(ssRes / ssTot);
        return (v < 0.0f) ? 0.0f : v;
    }

    // ------------------------------------------------------------
    // AUTO — meilleur r² parmi les modèles « sûrs »
    // ------------------------------------------------------------
    /**
     * @brief Teste les modèles linéarisés (0,1,2,3,5,6) et retient le meilleur r².
     *
     * Les modes NLS (4, 7, 8) sont EXCLUS de l'auto : ils sont lents sur
     * ESP8266 et peuvent diverger selon l'amorçage. À appeler explicitement.
     *
     * @param bestMode Reçoit le mode retenu
     * @param p0,p1,p2 Reçoivent les paramètres du modèle retenu
     * @return meilleur r²
     */
    static float autoFit(const float* x, const float* y, uint8_t n,
                         Mode& bestMode, float& p0, float& p1, float& p2) {
        bestMode = LIN; p0 = p1 = p2 = 0.0f;
        float bestR2 = -1.0f;
        float a, b, c, r2v;

        r2v = fitLinear(x, y, n, a, b);
        if (r2v > bestR2) { bestR2 = r2v; bestMode = LIN; p0 = a; p1 = b; p2 = 0.0f; }

        r2v = fitLog(x, y, n, a, b);
        if (r2v > bestR2) { bestR2 = r2v; bestMode = LOG; p0 = a; p1 = b; p2 = 0.0f; }

        r2v = fitExp(x, y, n, a, b);
        if (r2v > bestR2) { bestR2 = r2v; bestMode = EXP; p0 = a; p1 = b; p2 = 0.0f; }

        r2v = fitPow(x, y, n, a, b);
        if (r2v > bestR2) { bestR2 = r2v; bestMode = POW; p0 = a; p1 = b; p2 = 0.0f; }

        r2v = fitInv(x, y, n, a, b);
        if (r2v > bestR2) { bestR2 = r2v; bestMode = INV; p0 = a; p1 = b; p2 = 0.0f; }

        if (n >= 3) {
            r2v = fitPoly2(x, y, n, a, b, c);
            if (r2v > bestR2) { bestR2 = r2v; bestMode = POLY2; p0 = a; p1 = b; p2 = c; }
        }
        return (bestR2 < 0.0f) ? 0.0f : bestR2;
    }

    static const char* modeName(uint8_t m) {
        switch (m) {
            case LIN:     return "Lineaire";
            case LOG:     return "Logarithmique";
            case EXP:     return "Exponentielle";
            case POW:     return "Puissance";
            case POW_NLS: return "Puissance NLS";
            case INV:     return "Reciproque";
            case POLY2:   return "Polynome 2";
            case SIN:     return "Sinusoidal";
            case LOGIT:   return "Logistique";
            default:      return "Auto";
        }
    }
};

#endif // STATISTIQUES_H
