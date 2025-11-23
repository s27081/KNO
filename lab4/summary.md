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
Base Model Validation Accuracy: 0.9166666865348816
Base Model Validation Loss: 0.4205664396286011

### Testowane parametry:
- units: [8, 16, 32]
- dopout: 0-0.6
- learning-rate: 1e-3, 1e-2

**Metoda:**
RandomSearch
max_trials=5

**Wynik modelu ze zmienionymi parametrami:**
Hyperparameters Model Validation Accuracy 1.0
Hyperparameters Model Validation Loss 0.10930854827165604

**z parametrami:**
- units: 32
- dropout: 0.1
- learning-rate: 0.0083

**Macierz pomyłek**:
```bash
[[15  0  0]
 [ 0 14  0]
 [ 0  0  7]]
```

### Wnioski:
Optymalizowany model wykazał lepszą dokładność od bazowego mimo wysokiej dokładności modelu bazowego.

