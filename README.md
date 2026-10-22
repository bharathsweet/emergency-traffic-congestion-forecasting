# Emergency Dispatch & Traffic Congestion Forecasting

## Overview
This project uses an LSTM (Long Short-Term Memory) neural network to forecast traffic congestion from traffic and road-condition records. It is a prototype intended to support analysis for emergency dispatch planning.

## Model and architecture
- Framework: TensorFlow / Keras
- Model: Sequential LSTM neural network
- Input sequence: previous 24 hourly observations
- Layer 1: LSTM with 32 units (`return_sequences=True`)
- Dropout: 0.2
- Layer 2: LSTM with 16 units
- Dense hidden layer: 16 units with ReLU activation
- Output layer: 1 unit for congestion-index prediction
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)
- Metrics: Mean Absolute Error (MAE)
- Early stopping: monitors validation loss, patience 4

## Data workflow
1. Upload a ZIP file containing a CSV dataset in Google Colab.
2. Read up to 10,000 CSV records.
3. Parse timestamps and congestion index; remove rows missing either field.
4. Fill missing numeric values with medians and missing categorical values with `Unknown`.
5. One-hot encode categorical features.
6. Group observations by hour and calculate hourly averages.
7. Scale features using training-period data only.
8. Train on the first 80% of hourly observations and evaluate on the remaining 20%.
9. Calculate MAE and RMSE, plot actual versus predicted congestion, and forecast the next interval.

## Requirements
See `requirements.txt`. The code is designed to run in Google Colab because it uses `google.colab.files.upload()`.

## How to run in Google Colab
1. Open https://colab.research.google.com/
2. Select **File → Upload notebook** if using a notebook, or create a new notebook and paste `emergency_traffic_lstm.py` into code cells.
3. Run the code.
4. When prompted, upload a ZIP file containing the source CSV.
5. Wait for training and evaluation to finish.
6. The script creates `traffic_predictions.csv` and `emergency_traffic_lstm.keras` in the Colab runtime. Download them from the Files panel if you want to keep them.

## Dataset format
The ZIP file should contain a CSV with a `.csv` extension and columns including `Timestamp` and `Congestion_Index`. Other supported feature columns are listed in the code. The script selects at most 10,000 rows before preprocessing.

## Outputs
- `traffic_predictions.csv`: actual and predicted congestion values for test timestamps.
- `emergency_traffic_lstm.keras`: trained Keras model.

## Important note
This is a forecasting prototype. Its usefulness depends on dataset quality, time coverage, and evaluation results. It should not be used as the sole basis for real-world ambulance routing or emergency dispatch decisions.
