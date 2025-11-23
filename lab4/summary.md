# LAB 4

#### Model Bazowy
Dense(8, activation="relu")
Dropout(0.3, name="dropout")
Dense(3, activation="softmax")

**Paramtery:**
Optimizer: Adam(lr=default(0.001))
Epochs: 50
Batch_size: 15
**Wynik:**
- Serwer:
    - Accuracy: 0.9166666865348816
    - Loss: 0.4205664396286011
- Lokalnie:
    - Accuracy: 0.9166666865348816
    - Loss: 0.2378643900156021

### Testowane parametry:
- units: [8, 16, 32]
- dopout: 0-0.6
- learning-rate: 1e-3, 1e-2

**Metoda:**
RandomSearch
max_trials=5

**Wynik modelu ze zmienionymi parametrami:**
- Serwer:
    - Accuracy 1.0
    - 0.10930854827165604
- Lokalnie:
  - Accuracy 0.9722222089767456
  - Loss 0.24113532900810242

**z parametrami:**
- Serwer:
    - units: 32
    - dropout: 0.1
    - learning-rate: 0.0083
- Lokalnie:
    - units: 16
    - dropout: 0.0
    - learning-rate: 0.0069


**Macierz pomyłek**:

Serwer:
```bash
[[15  0  0]
 [ 0 14  0]
 [ 0  0  7]]
```
Lokalnie:
```bash
[[15  0  0]
 [ 1 13  0]
 [ 0  0  7]]
```
### Wnioski:
- Optymalizowany model wykazał lepszą dokładność od bazowego mimo wysokiej dokładności modelu bazowego.
- Czas uczenia i przwidywania oraz wyniki są bardzo podobne na serwerze oraz lokalnym środowisku.
