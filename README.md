# **Isolation Forest for Anomaly Detection**

This project implements an **Isolation Forest** model to detect anomalies in preprocessed time-series data. The dataset consists of road accelerometry readings, and the goal is to identify anomalous patterns (e.g., road distress or unusual activity).

---

## **Project Workflow**

### **1. Preprocessing**
The dataset is preprocessed before being fed into the model:
- **Magnitude Calculation**:
  - `acc_magnitude`: Magnitude of accelerometer readings calculated as `sqrt(ax^2 + ay^2 + az^2)`.
  - `gyro_magnitude`: Magnitude of gyroscope readings calculated similarly.
- **Low-Pass Filtering**:
  - Removes high-frequency noise using a low-pass Butterworth filter with a cutoff frequency of 10 Hz.
- **Feature Extraction**:
  - Features include `ax_filtered`, `ay_filtered`, `az_filtered`, and `acc_magnitude_filtered`.
- **Epoch Creation**:
  - The dataset is segmented into overlapping epochs for better analysis.

### **2. Isolation Forest Model**
The **Isolation Forest** is a tree-based anomaly detection algorithm that isolates data points to identify anomalies. It works by randomly splitting the dataset and measuring how quickly specific points can be separated.

#### **Key Parameters**:
- `n_estimators`: Number of trees in the forest (default: 100).
- `contamination`: Proportion of expected anomalies (default: 0.1).
- `max_samples`: Number of samples used to build each tree (default: 'auto').
- `max_features`: Number of features used in each split (default: 1.0).

#### **Steps**:
1. **Train-Test Split**:
   - The dataset is split into training (80%) and testing (20%) subsets.
2. **Data Normalization**:
   - Features are standardized using `StandardScaler` to ensure uniform scale.
3. **Training the Model**:
   - The Isolation Forest model is trained on the normalized training data.
4. **Anomaly Detection**:
   - Scores are generated using `decision_function`.
   - Data points below a threshold (e.g., 5th percentile of test scores) are classified as anomalies.

### **3. Validation**
Validation involves verifying flagged anomalies:
- **Extract Anomalies**:
  - Anomalies are identified in the test set based on the anomaly threshold.
- **Visualization**:
  - Visualize specific anomalies and their feature values.
- **Feature Inspection**:
  - Examine the feature values of anomalies for better understanding.

### **4. Fine-Tuning**
The model's performance is fine-tuned by:
1. **Adjusting Contamination**:
   - Experiment with different contamination levels (e.g., 0.05, 0.1, 0.15).
2. **Threshold Adjustment**:
   - Use various percentiles (e.g., 1st, 5th, 10th) to determine the optimal anomaly threshold.

### **5. Further Analysis**
Detailed analysis is conducted to understand anomalies:
- **Correlation Heatmap**:
  - Displays relationships between features to identify patterns in anomalies.
- **Temporal Visualization**:
  - Anomaly scores are plotted over time to detect trends or spikes in unusual activity.

---

## **Project Files**

1. **`isolation_forest_model.py`**:
   - Main Python script implementing the Isolation Forest model.
   - Includes preprocessing, training, anomaly detection, validation, fine-tuning, and visualization.

2. **`isolation_forest_train_results.csv`**:
   - Contains anomaly scores, predictions, and labels for the training set.

3. **`isolation_forest_test_results.csv`**:
   - Contains anomaly scores, predictions, and labels for the test set.

---

## **How to Use**

### **1. Prerequisites**
Install the required Python libraries:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
