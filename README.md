# Water Quality Detection and Prediction Using Time Series Models

## Authors
Xinyi Li, Peike Cong

## Abstract
This project explores time series and deep learning models, including LSTM, RNN, XGBoost, and ARIMA, for predicting water quality parameters. The report compares their performance and suitability for capturing complex time-dependent patterns.

---

## Introduction
### Background and Motivation
Water quality is essential for human life and ecosystems. Key parameters like dissolved oxygen (DO) are critical for monitoring water health. Predicting DO levels is vital for effective water management due to the time-dependent and interdependent nature of water quality parameters.

### Objective
The study evaluates various models to predict dissolved oxygen levels in water, comparing their ability to leverage temporal patterns and other features of water quality data.

---

## Data Description
### Data Sources
- **Queensland Government Open Data Portal**
- **Kaggle**

### Features
- Timestamp
- Record Number
- Average Water Speed
- Average Water Direction
- Chlorophyll
- Temperature
- Dissolved Oxygen
- pH
- Salinity
- Specific Conductance
- Turbidity

### Preprocessing
1. **Handling Missing Values**: KNN imputation with `n_neighbors=5`.
2. **Timestamp Processing**:
   - Converted to UNIX format.
   - Removed duplicates.
   - Added `TimeDiff` to represent time differences.
3. **Sequence Data Creation**:
   - Sliding window method with sequence length of 8.
   - Data normalization.
4. **Train-Test Split**: 80:10:10 ratio for training, validation, and test sets.

---

## Methodology
### Models Used
- **ARIMA**: Determines optimal (p, d, q) parameters.
- **RNN**: Two layers with hidden size of 15, dropout of 0.2, and early stopping.
- **LSTM**: Enhanced memory retention, same configuration as RNN.
- **XGBoost**: Decision trees with 100 estimators, learning rate of 0.1, and early stopping.

### Evaluation Metric
- **Mean Squared Error (MSE)**

---

## Results
![image](https://github.com/user-attachments/assets/ecfba0f7-f7c2-4dec-b757-00a43367936f)

| Model   | Epoch | Train Loss (Last Epoch) | Validation Loss (Last Epoch) | Test Loss  |
|---------|-------|--------------------------|-------------------------------|------------|
| ARIMA   | -     | -                        | -                             | 3.8314     |
| RNN     | 92.4  | 0.0702±0.0008          | 0.1432±0.0038               | 0.1520±0.0086 |
| LSTM    | 97.45 | 0.0617±0.0003          | 0.1577±0.0200               | 0.1848±0.0554 |
| XGBoost | 38.0  | 0.0457±0.0            | 0.1468±0.0                 | 0.1524±0.0    |

---

## Discussion
### Key Findings
- **LSTM**: Improved performance with Tanh activation due to better handling of non-linear dynamics.
- **RNN**: Faster convergence and effective for short-term dependencies.
- **XGBoost**: Strong for fitting data but limited for future forecasting due to lack of memory.

### Insights
- Simpler models (e.g., ARIMA) perform well on weakly correlated data.
- Neural networks (e.g., LSTM, RNN) excel in capturing complex temporal dependencies.

---

## References
1. H. B. Wang et al. (2018). *Online reliability time series prediction via CNN and LSTM*. Knowledge-Based Systems, 159, 132-147.
2. Enrique Sánchez et al. (2007). *Use of water quality index and DO deficit as watershed pollution indicators*. Ecological Indicators, 7(2), 315-328.
3. Katimon et al. (2018). *Modeling water quality using ARIMA*. Sustainable Water Resources Management, 4, 991-998.
4. Niazkar et al. (2024). *Applications of XGBoost in water resources engineering*. Environmental Modelling & Software, 174, 105971.
5. S. P. Roshan et al. (2023). *XGBoost-based techniques for water quality prediction*. ICCPCT, 364-369.
6. Y. Chen et al. (2017). *Recurrent neural networks for facial landmark detection*. Neurocomputing, 219, 26-38.

