# EMERGENCY DISPATCH & TRAFFIC CONGESTION FORECASTING
# Dataset: synthetic_traffic_benchmark_200k.csv
# Model: LSTM
# Records selected: up to 10,000

import zipfile
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from google.colab import files
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Upload ZIP dataset
uploaded = files.upload()
zip_name = next(iter(uploaded))

with zipfile.ZipFile(io.BytesIO(uploaded[zip_name])) as z:
    csv_name = next(
        n for n in z.namelist() if n.lower().endswith(".csv")
    )
    with z.open(csv_name) as f:
        df = pd.read_csv(f, nrows=10000)

print("Dataset loaded:", csv_name)
print("Selected records:", len(df))

# 2. Preprocess and sort by time
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df["Congestion_Index"] = pd.to_numeric(
    df["Congestion_Index"], errors="coerce"
)

df = df.dropna(subset=["Timestamp", "Congestion_Index"])
df = df.sort_values("Timestamp").drop_duplicates().reset_index(drop=True)

# 3. Select useful traffic-related features
numeric_features = [
    "Hour",
    "Month",
    "Is_Holiday",
    "Speed_Limit_kmh",
    "Visibility_m",
    "Recorded_Speed_kmh"
]

categorical_features = [
    "Day_of_Week",
    "Time_of_Day",
    "Area_Type",
    "Road_Type",
    "Weather",
    "Road_Condition",
    "Intersection_Type"
]

numeric_features = [c for c in numeric_features if c in df.columns]
categorical_features = [c for c in categorical_features if c in df.columns]

# Fill missing values
for c in numeric_features:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

for c in categorical_features:
    df[c] = df[c].fillna("Unknown").astype(str)

# Encode categorical data
features = pd.get_dummies(
    df[numeric_features + categorical_features],
    columns=categorical_features,
    dtype=float
)

features["Congestion_Index"] = df["Congestion_Index"].values
features["Timestamp"] = df["Timestamp"].values

# 4. Group observations into hourly averages
features["Timestamp"] = pd.to_datetime(features["Timestamp"])
features["Hour_Group"] = features["Timestamp"].dt.floor("h")

hourly = (
    features.drop(columns=["Timestamp"])
    .groupby("Hour_Group")
    .mean()
    .sort_index()
)

hourly = hourly.dropna()

target = "Congestion_Index"

print("Hourly records:", len(hourly))
print("Hourly data preview:")
print(hourly.head())

# 5. Normalize features without using test data
split = int(len(hourly) * 0.8)

if split <= 24 or len(hourly) - split < 1:
    raise ValueError("Not enough hourly observations for LSTM.")

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(hourly.iloc[:split])
test_scaled = scaler.transform(hourly.iloc[split:])

target_index = hourly.columns.get_loc(target)

# 6. Create sequences using previous 24 hourly observations
SEQ_LEN = 24

combined = np.vstack([train_scaled, test_scaled])

def create_sequences(data, start, end, seq_len, target_idx):
    X, y = [], []
    for i in range(start, end):
        X.append(data[i-seq_len:i, :])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(
    combined, SEQ_LEN, split, SEQ_LEN, target_index
)

X_test, y_test = create_sequences(
    combined, split, len(combined), SEQ_LEN, target_index
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# 7. Build LSTM model
tf.keras.utils.set_random_seed(42)

model = Sequential([
    Input(shape=(SEQ_LEN, X_train.shape[2])),
    LSTM(32, return_sequences=True),
    Dropout(0.2),
    LSTM(16),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# 8. Train model
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.1,
    shuffle=False,
    callbacks=[early_stop],
    verbose=1
)

# 9. Predict on unseen test data
pred_scaled = model.predict(X_test, verbose=0).flatten()

# Reverse normalization for the target column
def inverse_target(values):
    return (
        np.asarray(values) - scaler.min_[target_index]
    ) / scaler.scale_[target_index]

actual = inverse_target(y_test)
predicted = inverse_target(pred_scaled)

# 10. Calculate MAE and RMSE
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

print("\\nFINAL MODEL RESULTS")
print("-------------------")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")

# 11. Plot actual vs predicted congestion
test_times = hourly.index[split:]

plt.figure(figsize=(14, 5))
plt.plot(test_times, actual, label="Actual Congestion")
plt.plot(test_times, predicted, label="Predicted Congestion")
plt.title("Actual vs Predicted Traffic Congestion")
plt.xlabel("Time")
plt.ylabel("Congestion Index")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 12. Plot training and validation loss
plt.figure(figsize=(9, 4))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("LSTM Training Performance")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

# 13. Forecast the next observed hourly interval
latest_sequence = combined[-SEQ_LEN:].reshape(
    1, SEQ_LEN, combined.shape[1]
)

next_scaled = model.predict(latest_sequence, verbose=0)[0, 0]
next_congestion = inverse_target([next_scaled])[0]

print("\\nNEXT-HOUR CONGESTION FORECAST")
print("Last dataset hour:", hourly.index[-1])
print("Forecast congestion index:", round(float(next_congestion), 2))

# 14. Save results
results = pd.DataFrame({
    "Actual_Congestion": actual,
    "Predicted_Congestion": predicted
}, index=test_times)

results.to_csv("traffic_predictions.csv")
model.save("emergency_traffic_lstm.keras")

print("\\nSaved traffic_predictions.csv")
print("Saved emergency_traffic_lstm.keras")
