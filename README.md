# Artificial Intelligence Projects (TPs-IA)

This repository contains various Artificial Intelligence projects developed for academic purposes, covering fuzzy logic, clustering, genetic algorithms, neural networks, and LSTM analysis.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Projects](#projects)
  - [TP1 - Fuzzy Logic Systems](#tp1---fuzzy-logic-systems)
  - [Clustering - K-Means Implementation](#clustering---k-means-implementation)
  - [Genetic Algorithms](#genetic-algorithms)
  - [Neural Networks](#neural-networks)
  - [LSTM Networks Analysis](#lstm-networks-analysis)

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/NahuelBarriga/TPs-IA.git
cd TPs-IA
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install numpy matplotlib scikit-fuzzy scikit-learn pandas-datareader keras tensorflow
```

Individual package requirements:
- `numpy`: Numerical computing
- `matplotlib`: Plotting and visualization
- `scikit-fuzzy`: Fuzzy logic operations
- `scikit-learn`: Machine learning utilities (preprocessing, metrics)
- `pandas-datareader`: Financial data retrieval (for LSTM project)
- `keras`: Neural network library
- `tensorflow`: Deep learning framework (backend for Keras)

## Projects

### TP1 - Fuzzy Logic Systems

This project implements fuzzy logic inference systems for student grade evaluation.

**Location:** `TP1/`

**Files:**
- `mamdani.py`: Mamdani fuzzy inference system
- `Sugeno.py`: Sugeno fuzzy inference system
- `samplesVDA*.txt`: Sample data files

**Description:**
Implements fuzzy logic systems that evaluate student performance based on exam grades and concept grades to determine the final outcome (recursa/fail, habilita/pass, or promociona/promote).

**How to run:**

Mamdani system:
```bash
cd TP1
python mamdani.py
```

Sugeno system:
```bash
cd TP1
python Sugeno.py
```

**Expected output:** 
- Graphical visualization of membership functions
- Fuzzy inference results showing grade classifications
- Final grade determination based on input values

### Clustering - K-Means Implementation

This project implements the K-Means clustering algorithm from scratch.

**Location:** `Clustering/`

**Files:**
- `K-means.py`: Main K-Means implementation
- `prueba.py`: Alternative/test implementation
- `datos/datos.txt`: Sample dataset
- `samplesVDA1.txt`: Additional sample data

**Description:**
Custom implementation of the K-Means clustering algorithm that groups data points into clusters based on their similarity. The algorithm iteratively assigns data points to clusters and updates cluster centroids.

**How to run:**
```bash
cd Clustering
python K-means.py
```

**Expected output:**
- Visualization of data points colored by cluster assignment
- Cluster centroids
- Convergence statistics

### Genetic Algorithms

This project demonstrates genetic algorithm implementations for optimization problems.

**Location:** `Alg geneticos/`

**Files:**
- `AG_ej1.py`: Genetic algorithm for function optimization
- `AG_ProfeMati.py`: Genetic algorithm for image reconstruction

**Description:**
- `AG_ej1.py`: Uses a genetic algorithm to find the maximum of the function f(x) = 300 - (x - 15)^2 using binary encoding, selection, crossover, and mutation operations.
- `AG_ProfeMati.py`: Applies genetic algorithms to reconstruct a randomly generated binary image through evolutionary processes.

**How to run:**

Function optimization:
```bash
cd "Alg geneticos"
python AG_ej1.py
```

Image reconstruction:
```bash
cd "Alg geneticos"
python AG_ProfeMati.py
```

**Expected output:**
- Progress of generations showing fitness improvement
- Final optimized solution
- For image reconstruction: visual comparison of original and reconstructed images

### Neural Networks

This project contains neural network implementations and experiments.

**Location:** `Redes Neuronales/`

**Files:**
- `RN_ej1`: Simple neural network example for stock price prediction
- `TpFinal.ipynb`: Jupyter notebook with final project

**Description:**
Implements neural networks for time series prediction, specifically working with stock market data (SPY ticker) to predict closing prices.

**How to run:**

For the Python script:
```bash
cd "Redes Neuronales"
python RN_ej1
```
Note: The RN_ej1 file does not have a .py extension but is a valid Python script.

For the Jupyter notebook:
```bash
cd "Redes Neuronales"
jupyter notebook TpFinal.ipynb
```

**Expected output:**
- Historical price data visualization
- Neural network training progress
- Prediction results and accuracy metrics

### LSTM Networks Analysis

This project uses Long Short-Term Memory (LSTM) networks for time series analysis.

**Location:** `Analisis Redes LSTM/`

**Files:**
- `TpFinal.ipynb`: Jupyter notebook with LSTM implementation
- `datos.txt`: Dataset for analysis
- `Readme.txt`: Project notes

**Description:**
Implements LSTM neural networks for analyzing and predicting time series data, particularly financial market data. LSTMs are well-suited for sequential data due to their ability to capture long-term dependencies.

**How to run:**
```bash
cd "Analisis Redes LSTM"
jupyter notebook TpFinal.ipynb
```

**Expected output:**
- Data preprocessing and normalization steps
- LSTM model training visualization
- Prediction results and model evaluation metrics

## Notes

- Some projects require internet connectivity to download financial data
- Execution times may vary depending on dataset size and hardware specifications
- The Jupyter notebooks provide interactive exploration and detailed explanations of the implementations
- Make sure all data files (.txt) are in their respective directories before running the scripts

## License

This repository is for academic purposes. The intellectual property rights of any associated reports belong to their respective authors.