# Microsoft Stock Forecasting using LSTM

This project predicts the future stock prices of Microsoft (MSFT) using a Long Short-Term Memory (LSTM) neural network. It uses historical stock price data to train the model and forecast future stock movements. The project is built with Streamlit for an interactive web interface, allowing users to train the model or load a pre-trained model to predict future prices.

## Features
- Fetches historical stock data for Microsoft from Yahoo Finance.
- Preprocesses the stock data and applies MinMax scaling.
- Builds an LSTM model for stock price prediction.
- Option to train a new model or load a pre-trained `.h5` model.
- Saves the trained LSTM model as an `.h5` file.
- Predicts and visualizes future stock prices using Streamlit.

## Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python, TensorFlow, Keras
- **Data Source**: Yahoo Finance (via `yfinance` library)

Usage
After running the app, you will see the Streamlit interface in your browser.
Use the sidebar to:
Enter the stock ticker (MSFT is pre-filled by default).
Optionally upload a pre-trained LSTM model (.h5 file).
The app will automatically fetch the Microsoft stock data and display a chart.
You can either:
Train a new LSTM model on the data.
Load a pre-trained model to predict future prices.
The app will display the predicted stock prices along with the actual prices on a line chart.
Model Architecture
The LSTM model is built using TensorFlow and Keras. It has the following structure:

LSTM Layer 1: 50 units, returning sequences.
LSTM Layer 2: 50 units, no return sequences.
Dense Layer 1: 25 units.
Dense Layer 2: 1 unit (output layer).
The model is trained using the Adam optimizer and mean_squared_error as the loss function.

Example Prediction

File Structure
app.py: Main Python script for the Streamlit app.
README.md: Documentation for the project.
requirements.txt: List of dependencies.
Future Improvements
Add support for other stock tickers besides Microsoft.
Implement hyperparameter tuning for the LSTM model.
Allow users to select different time periods for training and prediction.
Enhance visualization with more detailed metrics and charts.