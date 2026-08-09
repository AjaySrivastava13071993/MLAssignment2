# Mobile Price Classification - Machine Learning Assignment 2

**BITS Pilani Work Integrated Learning Programmes Division**  
**Course:** Machine Learning (M.Tech AIML / DSE)  
**Author:** Ajay Srivastava  
**Submission Deadline:** 18-Aug-2026  

---

## 1. Problem Statement
The objective of this assignment is to develop an end-to-end Machine Learning pipeline for **Mobile Price Classification**. The problem predicts the target variable `price_range`, which categorizes mobile devices into four distinct price tiers based on hardware specifications:
* `0`: Low Cost
* `1`: Medium Cost
* `2`: High Cost
* `3`: Very High Cost

---

## 2. Dataset Description

**Dataset Name:** Mobile Price Classification  
**Source:** [Kaggle - iabhishekofficial/mobile-price-classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)  
**Total Instances:** 2,000 (satisfies minimum requirement of 500)  
**Total Features:** 20 (satisfies minimum requirement of 12)  
**Target Variable:** `price_range` (4 classes: 0, 1, 2, 3)

### Feature Details:
* **Continuous / Discrete Numerical Features**:
  * `battery_power`: Battery capacity (mAh)
  * `clock_speed`: Microprocessor execution speed (GHz)
  * `fc`: Front camera megapixels
  * `int_memory`: Internal memory (GB)
  * `m_dep`: Mobile depth (cm)
  * `mobile_wt`: Weight of phone (g)
  * `n_cores`: Processor cores
  * `pc`: Primary camera megapixels
  * `px_height`: Pixel resolution height
  * `px_width`: Pixel resolution width
  * `ram`: Random Access Memory (MB)
  * `sc_h`: Screen height (cm)
  * `sc_w`: Screen width (cm)
  * `talk_time`: Max battery talk time (hours)
* **Binary Categorical Features (0 or 1)**:
  * `blue` (Bluetooth), `dual_sim` (Dual SIM), `four_g` (4G), `three_g` (3G), `touch_screen` (Touch Screen), `wifi` (Wi-Fi)

---

## 3. GitHub Repository Link
**Repository URL:** [https://github.com/AjaySrivastava13071993/MLAssignment2](https://github.com/AjaySrivastava13071993/MLAssignment2)

---

## 3b. Live Streamlit App Link
**Live App URL:** [https://mlassignment2-7tmja62s3eqjtnbvbo6zig.streamlit.app/](https://mlassignment2-7tmja62s3eqjtnbvbo6zig.streamlit.app/)

---

## 4. Models Used

Five classification algorithms were trained and fine-tuned using **5-Fold Cross Validation (`GridSearchCV`)** with a custom random state seed (`1307`). Features were standardized using `StandardScaler` before training.

1. **Logistic Regression**: Best parameters: `C=100.0`, `solver='lbfgs'`
2. **Decision Tree Classifier**: Best parameters: `criterion='entropy'`, `max_depth=6`, `min_samples_split=2`
3. **K-Nearest Neighbors (KNN)**: Best parameters: `metric='manhattan'`, `n_neighbors=21`, `weights='distance'`
4. **Gaussian Naive Bayes**: Best parameter: `var_smoothing=0.0687`
5. **Random Forest Classifier** (Ensemble): Best parameters: `criterion='entropy'`, `max_depth=None`, `min_samples_split=6`, `n_estimators=200`

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9800 | 0.9995 | 0.9801 | 0.9800 | 0.9800 | 0.9734 |
| Decision Tree | 0.8300 | 0.9542 | 0.8309 | 0.8300 | 0.8301 | 0.7735 |
| kNN | 0.6750 | 0.8540 | 0.6757 | 0.6750 | 0.6753 | 0.5667 |
| Naive Bayes | 0.7900 | 0.9495 | 0.7960 | 0.7900 | 0.7921 | 0.7206 |
| Random Forest (Ensemble) | 0.8875 | 0.9805 | 0.8903 | 0.8875 | 0.8885 | 0.8503 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Achieved the highest accuracy (98.00%) and near-perfect AUC (0.9995). Standardized features reveal strong linear relationships between RAM/pixel dimensions and price tier, allowing linear hyperplanes to cleanly separate all 4 classes. |
| Decision Tree | Achieved 83.00% accuracy. Provides a clear, interpretable rule-based structure. However, individual trees show higher variance than ensemble methods, and some misclassification occurs at the class boundaries between adjacent price tiers. |
| kNN | Achieved the lowest accuracy (67.50%). Distance-based classification suffers from the curse of dimensionality across 20 features, and the presence of binary categorical features dilutes the quality of Euclidean/Manhattan distance calculations. |
| Naive Bayes | Achieved 79.00% accuracy. The Gaussian independence assumption restricts performance since many mobile features are correlated (e.g., RAM and price are strongly co-dependent). Despite this, the model shows a relatively high AUC of 0.9495. |
| Random Forest (Ensemble) | Achieved 88.75% accuracy and AUC of 0.9805. By aggregating 200 decision trees, the ensemble effectively reduces variance and captures non-linear feature interactions, making it the second-best performer on this dataset. |
| **Overall Winner for your dataset?** | **Logistic Regression** — Achieves 98.00% accuracy with a near-perfect AUC of 0.9995, outperforming all other classifiers by a significant margin on the Mobile Price Classification dataset. |

---

## 5. Streamlit Dashboard Features
The Streamlit app (`app.py`) includes a custom 4-tab interactive interface:
1. **Live Single Device Predictor**: Interactive sliders/number inputs for instant real-time hardware spec predictions.
2. **Batch CSV Evaluator**: File upload widget for batch CSV evaluation, confusion matrix heatmaps, classification reports, and downloadable prediction CSVs.
3. **Model Performance Benchmark**: Graphical metric benchmark comparisons across all 5 models.
4. **Dataset & Problem Overview**: Detailed feature specifications and problem context.

---

## 6. How to Run Locally

```bash
# Clone the repository
git clone https://github.com/AjaySrivastava13071993/MLAssignment2.git
cd MLAssignment2

# Install dependencies
pip install -r requirements.txt

# Run Streamlit Web Application
streamlit run app.py
```
