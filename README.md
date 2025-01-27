#  **AIProjectActiGence**

<h2><u>Overview</u></h2>

This project leverages Long Short-Term Memory (LSTM) neural networks to analyze and make predictions on financial data. Using a combination of Python libraries and technical indicators, it demonstrates a robust pipeline for processing, visualizing, and forecasting stock prices.

<b>Features</b>

**Data Collection**: Stock price data is fetched from Yahoo Finance using the yfinance library.

**Preprocessing**: Prepares and cleans financial time-series data.

**Visualization**: Uses matplotlib for visualizing trends and patterns in the data.

**Modeling**: Implements LSTM neural networks for sequence prediction using TensorFlow.

**Evaluation**: Provides insights into model performance using various metrics.


**Required Libraries**

Ensure you have the following libraries installed:

- numpy

- pandas

- matplotlib

- tensorflow

- yfinance

- datetime


**Usage**

- Clone the repository and navigate to the project directory.

- Open the provided Jupyter Notebook AI-LSTM-Technical.ipynb.

- Follow the steps outlined in the notebook:

  - Data fetching

  - Preprocessing

  - Model training

  - Visualization and evaluation


 **Key Components**


 **Data Collection**:
 Utilizes the yfinance library to retrieve historical stock price data. You can customize the ticker symbol and time range.

**Preprocessing**:
Handles missing data, normalization, and transformation into a format suitable for LSTM input.

**Modeling**:
Builds and trains an LSTM neural network using TensorFlow. The model is designed to capture temporal dependencies in financial data.

**Visualization**:
Employs matplotlib to plot:
- Stock price trends
- Model predictions vs. actual values


**Results**

The notebook demonstrates:

- Near Accurate predictions on test datasets.

- Insights into the performance of the LSTM model.


