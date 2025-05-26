
# Seoul-Bike-Sharing-Demand-Prediction

## 📌 Project Overview
This project predicts the demand for bike-sharing services in Seoul using machine learning algorithms. It utilizes historical data and environmental factors to provide accurate predictions that can aid in resource optimization and planning.

---

## 📊 Dataset and Features
The dataset contains historical bike-sharing demand data, including:
- **Date and Time**
- **Weather Data**: Temperature, Humidity, Wind Speed, Solar Radiation, Visibility, Rainfall, and Snowfall.
- **Seasonal and Calendar Information**: Season, Holidays, Functioning Days.
- **Derived Features**: Weekday names, specific months, and years.

---

## 🛠️ Project Workflow
1. **Import Packages**: Required libraries for data manipulation, visualization, and machine learning.
2. **Load Data**: Read and explore the dataset.
3. **Data Cleaning**: Handle missing values and outliers to prepare the dataset for analysis.
4. **Feature Engineering**: Add new features like weekdays, seasons, etc., to enhance model performance.
5. **Exploratory Data Analysis (EDA)**: Visualize relationships between variables and demand.
6. **Remove Multicollinearity**: Identify and handle highly correlated features.
7. **Encoding**: Convert categorical features into numeric format.
8. **Split Data for Training & Testing**: Divide the data into training and testing sets.
9. **Scaling**: Standardize numeric features to improve model accuracy.
10. **Training ML Models**: Build and train multiple machine learning models, including:
    - Linear Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - XGBRegressor (primary model)
11. **Model Evaluation**: Evaluate models using R² score and other metrics.
12. **Train Multiple Models**: Compare results from 8 different models.
13. **Visualize Model Predictions**: Plot and compare predictions vs. actual demand.
14. **Save ML Model**: Save the trained XGBRegressor model for future use.
15. **Dump Scaling Parameters**: Save the scaling parameters to ensure consistency during inference.
16. **Load Model**: Reload the saved model for deployment.
17. **Load Scaling Parameters**: Reload the saved scaler for new data.
18. **User Input**: Accept custom input for real-time predictions.
19. **Convert User Data**: Transform user input into a format consumable by the model.
20. **Deployment Prediction**: Generate predictions for bike demand based on user input.

---

## 🔑 Key Features
- **Accurate Demand Prediction**: Achieved an R² score of 0.947 using XGBRegressor.
- **User Input for Predictions**: Interactive prediction system using custom inputs.
- **Reusable Pipeline**: Saved model and scaler for deployment.

---

## 🧑‍💻 Technologies Used
- **Programming Language:** Python
- **Libraries:**  
  - `pandas`, `numpy` for data processing  
  - `matplotlib`, `seaborn` for visualization  
  - `scikit-learn`, `xgboost` for machine learning  
  - `pickle` for saving/loading models and parameters  

---

## 🚀 How to Run the Project
1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/your-username/Seoul-Bike-Sharing-Demand-Prediction.git
   cd Seoul-Bike-Sharing-Demand-Prediction
