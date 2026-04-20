# Uncertainty-Aware Blood Glucose Prediction using Bayesian BiGRU

## Overview
This project develops a Bayesian deep learning model for predicting blood glucose levels in Type 1 Diabetes patients.  
Unlike traditional models, it provides both predictions and uncertainty estimates, enabling safer and more reliable decision-making.

---

## Key Features
- Bidirectional GRU (BiGRU) for time-series prediction  
- Attention Mechanism for identifying important time steps  
- Bayesian Output Layer (mean and variance)  
- Uncertainty Quantification:
  - Aleatoric (data noise)
  - Epistemic (model uncertainty)
- LINEX Loss for asymmetric clinical risk  
- Well-calibrated predictions (~96% coverage)

---

## Model Architecture
- 3-layer Bidirectional GRU  
- Attention layer  
- BayesianLinear output:
  - Mean (prediction)
  - Variance (uncertainty)

---

## Results
- RMSE: ~22 mg/dL  
- MAE: ~15.49 mg/dL  
- R² Score: ~0.94  
- Coverage: ~96.49% within 95% interval  

---

## Uncertainty Analysis
- Aleatoric uncertainty dominates  
- Epistemic uncertainty is low  
- Uncertainty aligns with prediction error (good calibration)  

---

## Statistical Validation
- Fisher Information: ~18.897  
- CRLB: ~0.0529  
- UMVUE achieves CRLB (optimal estimator)  

---

## Loss Function Comparison
- MSE / RMSE: symmetric error  
- MAE: robust to outliers  
- LINEX: asymmetric clinical risk  

---

## Dataset
OhioT1DM Dataset (Kaggle):  
https://www.kaggle.com/datasets/ryanmouton/ohiot1dm/data  

---

## Reference
Bayesian Neural Network for Uncertainty-Aware Blood Glucose Prediction for Type 1 Diabetes  
https://share.google/wBmJRvxiuvFX2CNwg  

---

## Tech Stack
- Python  
- PyTorch  
- NumPy, Pandas  
- Matplotlib  

---

## Applications
- Glucose prediction  
- Hypoglycemia risk detection  
- Clinical decision support  

