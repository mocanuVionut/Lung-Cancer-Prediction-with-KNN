#  Lung Cancer Prediction with KNN (MATLAB)

This project implements a **K-Nearest Neighbors (KNN)** classifier in **MATLAB** to predict lung cancer risk based on patient data. The dataset contains patient attributes and symptoms, and the model classifies individuals into two categories: `LUNG_CANCER` = YES / NO.

Repo: `Lung-Cancer-Prediction-with-KNN`

---

##  Features

-  **Reads dataset (`lung-cancer.data`)** with preprocessing
-  **Handles missing values** by replacing `?` with `NaN` and removing incomplete rows
-  **Splits data into train/test sets** (75% training, 25% testing)
-  **Trains KNN classifier** with:
  - `k = 5` neighbors
  - Euclidean distance metric
-  **Evaluates performance** with accuracy and confusion matrix

---

##  Dataset

- File: `lung-cancer.data`
- Column 1: **Label** (`LUNG_CANCER`: 1 = YES, 2 = NO or equivalent)  
- Columns 2–end: **Features** (patient attributes & symptoms)

>  **Note:** The dataset is intended **only for academic and research purposes**. It must not be used for clinical or real medical decisions.

---

##  Methodology

1. **Data Loading**
   ```matlab
   filename = 'lung-cancer.data';
   opts = detectImportOptions(filename, 'FileType', 'text', 'NumHeaderLines', 0);
   opts.MissingRule = 'fill';
   opts = setvartype(opts, 'char');  
   data = readtable(filename, opts);

2. **Preprocessing**
- Replace `?` with `NaN`  
- Convert all entries to `double`  
- Remove rows with missing values  

3. **Feature / Label Split**
```matlab
X = table2array(data(:, 2:end));  % Features
y = table2array(data(:, 1));      % Labels

4. **Train / Test Split**
```matlab
cv = cvpartition(size(X,1), 'HoldOut', 0.25);

5. **KNN Training**
```matlab
knnModel = fitcknn(XTrain, YTrain, 'NumNeighbors', 5, 'Distance', 'euclidean');

6. **Evaluation**

-  Predict on test set
-  Calculate accuracy
-  Display confusion matrix

```matlab
YPred = predict(knnModel, XTest);
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf("Test Accuracy: %.2f%%\n", accuracy * 100);

cm = confusionchart(YTest, YPred);
cm.Title = 'Confusion Matrix for KNN Lung Cancer Classification';
