# AI-Based Energy Consumption Forecasting: Viva Questions & Answers

This document provides simple, beginner-friendly explanations for common questions asked during college vivas, project presentations, and examinations.

---

### 1. What is Artificial Intelligence (AI)?
**Answer**: Artificial Intelligence refers to the simulation of human intelligence in machines. It enables computer systems to perform tasks that typically require human intelligence, such as recognizing patterns, making decisions, learning from experience, and solving complex problems.

---

### 2. What is Machine Learning (ML)?
**Answer**: Machine Learning is a branch of AI where computer algorithms learn patterns and relationships directly from historical data without being explicitly hard-coded with fixed rules. As more data is provided, the system's prediction accuracy improves.

---

### 3. What is Supervised Learning?
**Answer**: Supervised Learning is a category of Machine Learning where models are trained on **labeled data**. This means during training, the algorithm is provided with both the input features ($X$) and the correct ground-truth target outputs ($y$). The model learns a mathematical mapping function from inputs to outputs.

---

### 4. Why is this project a Regression problem and not a Classification problem?
**Answer**:
- **Classification** predicts discrete categories or classes (e.g., "High vs Low Demand", "Spam vs Not Spam").
- **Regression** predicts a continuous numerical quantity (e.g., predicting exact electricity demand: $125.4$ Mega Units).
Since energy consumption is a continuous numerical value, this is a **Regression** problem.

---

### 5. What is Energy Consumption Forecasting and why is it important?
**Answer**: Energy consumption forecasting is the process of predicting future electricity demand using past historical consumption records, seasonal calendar trends, and regional factors.  
**Importance**:
- Helps power grid operators balance electricity generation and demand.
- Prevents blackouts and grid overloading.
- Minimizes wasted electricity generation and reduces operational costs.
- Assists in planning renewable energy integration.

---

### 6. What is a "Feature" (Independent Variable, $X$)?
**Answer**: A feature is an input variable used by the machine learning model to make predictions. In our project, features include:
- Temporal features: `Month`, `Day`, `DayOfWeek`, `IsWeekend`, `Quarter`, `DayOfYear`.
- Geographic features: `latitude`, `longitude`, `States_Encoded`, `Regions_Encoded`.
- Historical signals: `Usage_Lag_1` (yesterday's consumption), `Usage_Lag_7` (last week same day), `Usage_Rolling_Mean_7` (7-day average).

---

### 7. What is the "Target Variable" (Dependent Variable, $y$)?
**Answer**: The target variable is the specific outcome we want the model to learn to predict. In this project, the target variable is **`Usage`** (Daily Energy Consumption in **Mega Units - MU**).

---

### 8. Why do we split data into Training and Testing sets?
**Answer**: If we evaluate a model on the same data it learned from, it might simply memorize the answers rather than generalizing to new data (overfitting). Splitting into training (80%) and testing (20%) allows us to objectively evaluate how well the model predicts **unseen future data**.

---

### 9. Why shouldn't we randomly shuffle data in time-series forecasting?
**Answer**: In time-series forecasting, time order matters. If we shuffle randomly, data from future dates will be placed into the training set to predict past dates in the test set. This creates **future data leakage** (the model "cheats" by knowing future trends). Instead, we use a **Chronological Split** (first 80% past days for training, last 20% future days for testing).

---

### 10. What is Linear Regression?
**Answer**: Linear Regression is a fundamental baseline algorithm that models a straight-line relationship between the input features and the target variable using the equation:
$$y = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$
It finds the line (or hyperplane) that minimizes the sum of squared differences between actual and predicted points.

---

### 11. What is a Decision Tree Regressor?
**Answer**: A Decision Tree breaks down a dataset into smaller subsets while simultaneously developing an associated tree structure. It makes decisions through simple if-else questions (e.g., *Is Month $\ge 5$?*, *Is State == Maharashtra?*) to assign average consumption values in the leaf nodes.

---

### 12. What is Random Forest Regressor?
**Answer**: Random Forest is an **ensemble** learning method based on **Bagging (Bootstrap Aggregation)**. It builds a forest of multiple independent decision trees (e.g., 100 trees), each trained on random subsets of data and features. The final prediction is the average of all individual tree predictions, which significantly reduces variance and prevents overfitting.

---

### 13. What is Gradient Boosting Regressor?
**Answer**: Gradient Boosting is an ensemble technique based on **Boosting**. Unlike Random Forest where trees are built independently, Gradient Boosting builds trees **sequentially**. Each new decision tree focuses specifically on correcting the residual errors made by the previous trees, leading to high predictive accuracy.

---

### 14. What is Mean Absolute Error (MAE)?
**Answer**: MAE measures the average absolute difference between the actual energy consumption and the predicted consumption:
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
**Interpretation**: An MAE of $6.1$ MU means our predictions are, on average, within $6.1$ Mega Units of the real energy usage.

---

### 15. What is Root Mean Squared Error (RMSE)?
**Answer**: RMSE is the square root of the average squared differences:
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
**Interpretation**: Because errors are squared before averaging, RMSE penalizes large errors much more heavily than MAE. A lower RMSE indicates fewer severe forecasting mistakes.

---

### 16. What is the $R^2$ Score (Coefficient of Determination)?
**Answer**: The $R^2$ score measures the proportion of total variance in the target variable that is successfully explained by the model:
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
- $R^2 = 1.0$: Perfect predictions.
- $R^2 = 0.0$: Model predicts no better than the simple mean.
- In our project, an $R^2 \approx 0.98$ means our model explains $98\%$ of the variation in daily electricity consumption.

---

### 17. Which model performed the best in your project and why?
**Answer**: **Gradient Boosting Regressor** performed the best with an $R^2$ score of $0.9837$ and the lowest RMSE ($14.84$ MU).  
**Why**: Gradient Boosting sequentially optimizes errors across non-linear seasonal variations and state-specific demand baselines, capturing complex interactions that simpler models cannot capture.

---

### 18. What is Overfitting and how did you prevent it?
**Answer**:
- **Overfitting** occurs when a model learns the noise and exact samples in the training data so well that it fails on new, unseen data.
- **How we prevented it**:
  1. Used ensemble algorithms (Random Forest & Gradient Boosting).
  2. Constrained tree depth (`max_depth=10`, `max_depth=5`) to prevent memorization.
  3. Evaluated models strictly on an unseen 20% future test set.

---

### 19. What is Data Leakage and how did you prevent it?
**Answer**: Data Leakage happens when information from outside the training dataset (or future information) is inadvertently leaked into the model during training.  
**How we prevented it**:
1. All lag and rolling features used backward shifts (`shift(1)`), ensuring features only look into the past.
2. We performed a chronological train-test split rather than a random shuffle.
3. Encoders were fit on training data and stored.

---

### 20. What are the limitations of this project?
**Answer**:
1. **Weather Data Absence**: Weather variables (temperature, humidity, precipitation) directly influence air-conditioning usage; including them would further improve accuracy.
2. **Daily Resolution**: The dataset is daily; hourly or 15-minute smart meter data would allow real-time peak-load forecasting.
3. **External Disruptions**: Unprecedented events (e.g., unexpected grid maintenance or lockdown anomalies) cannot be fully captured without external event markers.

---

### 21. How can this project be enhanced in the future?
**Answer**:
1. Integrate live weather API data (e.g., OpenWeatherMap temperature and humidity).
2. Add hourly or 15-minute resolution smart grid data.
3. Implement deep learning architectures (e.g., LSTM, Transformer) for multi-step ahead continuous forecasting.
4. Deploy the Streamlit application to the cloud with real-time automated grid alerts.
