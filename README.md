# Statistiques

**Statistiques et régressions pour l'embarqué** — header-only, zéro allocation
dynamique, ESP8266 / ESP32 / Arduino.

Auteur : Olivier FOURNET · Licence : GPL-3.0-only · Version : 1.0.0

---

## Pourquoi un portage et non le projet d'origine

Ce dépôt est le **portage Arduino** de [`Fo170/Statistiques`](https://github.com/Fo170/Statistiques),
qui est une **application Qt**.

L'amont ne compile pas sur microcontrôleur :

| Problème | Détail |
|---|---|
| `statsengine.cpp` | dépend de `QString` / `QVector` |
| `includes/*.hl` | fichiers **DOS de 1996** — macros `_()`, `#ifdef ORD` pour activer les fonctions, extension `.hl` |
| Précision | `double` là où un `float` suffit (et coûte 2× la place) |
| Mémoire | `QVector` alloue — interdit sur un ESP8266 (fragmentation du heap) |

**Ce portage conserve les algorithmes** (les 9 modèles de régression sont
identiques) en `float`, avec des **tableaux statiques de taille fixe**.
Aucune allocation dynamique, aucune dépendance hors `Arduino.h` et `math.h`.

---

## API

Toutes les méthodes sont **statiques** : aucun objet à construire, aucun état.

### Statistiques descriptives

| Méthode | Rôle |
|---|---|
| `mean(x, n)` | Moyenne |
| `variance(x, n)` | Variance |
| `stddev(x, n)` | Écart-type |
| `median(x, n)` | **Médiane** (robuste aux outliers) |
| `mad(x, n)` | **MAD** — écart absolu médian |
| `minMax(x, n, &vmin, &vmax)` | Bornes |
| `covariance(x, y, n)` | Covariance |
| `correlation(x, y, n)` | Corrélation |
| `sortAsc(v, n)` | Tri croissant sur place |

### Régressions (9 modèles)

| Méthode | Modèle |
|---|---|
| `fitLinear(x, y, n, ...)` | Linéaire `y = a·x + b` |
| `fitLog(x, y, n, ...)` | Logarithmique `y = a·ln(x) + b` |
| `fitExp(x, y, n, ...)` | Exponentiel `y = a·e^(b·x)` |
| `fitPow(x, y, n, ...)` | Puissance `y = a·x^b` |
| `fitInv(x, y, n, ...)` | Inverse `y = a/x + b` |
| `fitPoly2(x, y, n, ...)` | Polynôme de degré 2 |
| `eval(mode, x, a, b, c, ...)` | Évaluation d'un modèle ajusté |
| `r2(x, y, n, ...)` | Coefficient de détermination |
| `r2Poly2(x, y, n, ...)` | R² du polynôme de degré 2 |

---

## Installation

Aucune dépendance.

```ini
lib_deps =
    https://github.com/Fo170/Statistiques-Arduino.git#v1.0.0
```

Arduino IDE : copier le dossier dans `libraries/`.

---

## Utilisation

```cpp
#include <Statistiques.h>

float x[8] = {0, 1, 2, 3, 4, 5, 6, 7};
float y[8] = {1.1f, 3.0f, 5.2f, 6.9f, 9.1f, 11.0f, 13.2f, 14.8f};

void setup() {
    Serial.begin(115200);

    // Médiane et MAD : robustes jusqu'à 50 % de valeurs aberrantes
    float med = Statistiques::median(y, 8);
    float mad = Statistiques::mad(y, 8);

    // Régression linéaire
    float a, b, r2;
    Statistiques::fitLinear(x, y, 8, &a, &b, &r2);
    Serial.printf("y = %.3f x + %.3f  (R2 = %.4f)\n", a, b, r2);
}

void loop() {}
```

---

## Vérifications

- Compilation croisée **ESP8266 autonome** (projet PlatformIO jetable,
  `#include <Statistiques.h>` seul) → **SUCCESS**.
- 32 vérifications hôte du projet Coulométre (`tools/math_tests`) : **0 échec**.
- Aucun `new`, `malloc` ou `String` dans le header.

---

## Licence

GPL-3.0-only — Olivier FOURNET
