# Ensemble-Deep-Learning-Models-for-Stock-Price-Forecasting

Ensemble Deep Learning Models for Stock Price Forecasting
This repository contains the code and resources for the project "Ensemble Deep Learning Models for Stock Price Forecasting", which focuses on predicting stock prices in the Sri Lankan market using advanced deep learning techniques. The project combines Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) architectures in an ensemble framework for enhanced accuracy and robustness.

Overview
Stock price forecasting is a challenging task due to the complexity of financial markets. This project develops a hybrid model leveraging LSTM and GRU to capture both short-term and long-term dependencies in stock price movements. An ensemble approach is employed to combine the predictions from individual models, improving the system's overall reliability and performance.

Key Features:
Hybrid LSTM-GRU Model: Combines the strengths of LSTM and GRU for superior temporal pattern recognition.
Ensemble Learning: Reduces prediction variance and overfitting.
Hyperparameter Optimization: Uses techniques like grid search and Bayesian optimization for tuning.
Robust Evaluation: Employs metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
Dataset
The project uses 10 years of historical stock price data from the Colombo Stock Exchange (CSE) for the following companies:

Ceylon Tobacco
Sampath Bank
Dialog PLC
John Keells Holdings
Dilmah
Each dataset includes daily stock information such as Open, High, Low, Close, and Volume.

Preprocessing Steps:
Handle missing data using interpolation.
Normalize data for consistency.
Engineer features like moving averages and volatility indicators.
Split data into training (65%) and testing (35%) sets.
Methodology
The development process is divided into the following phases:

1. Model Development
LSTM Model: Captures long-term dependencies.
GRU Model: Efficiently handles short-term patterns.
Hybrid LSTM-GRU Model: Integrates LSTM and GRU for enhanced performance.
2. Ensemble Approach
Combines predictions using weighted averaging.
Improves robustness and reduces overfitting.
3. Hyperparameter Optimization
Fine-tuned parameters include:
Number of units
Learning rate
Batch size
Dropout rate
Optimizer
Epochs
Optimization techniques: Grid Search, Random Search, Bayesian Optimization.
Results
The Hybrid LSTM-GRU Model outperformed standalone LSTM and GRU models across all companies, providing the most accurate predictions.

Installation

Requirements:
Python 3.8 or higher
TensorFlow
NumPy
Pandas
Matplotlib
Scikit-learn

Acknowledgments
Department of Mathematics, Faculty of Science, University of Colombo
Colombo Stock Exchange (CSE) for the data
References to previous works and publications are included in the project report.
