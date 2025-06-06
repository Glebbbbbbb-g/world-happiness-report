import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setting page configuration
st.set_page_config(page_title="World Happiness Report Analysis", layout="wide")

# Loading the data
@st.cache_data
def load_data():
    data = pd.read_csv("../Downloads/2019.csv")
    return data

# Data description
st.title("World Happiness Report 2019 Analysis")
st.markdown("""
### Data Description
The dataset is from the 2019 World Happiness Report, containing happiness scores and related factors for various countries. The columns include:
- **Overall rank**: Rank of the country based on happiness score.
- **Country or region**: Name of the country or region.
- **Score**: Happiness score (target variable).
- **GDP per capita**: Contribution of GDP per capita to happiness.
- **Social support**: Contribution of social support to happiness.
- **Healthy life expectancy**: Contribution of healthy life expectancy to happiness.
- **Freedom to make life choices**: Contribution of freedom to happiness.
- **Generosity**: Contribution of generosity to happiness.
- **Perceptions of corruption**: Contribution of perceived corruption to happiness.
""")

# Loading and displaying data
data = load_data()
st.subheader("Dataset Preview")
st.dataframe(data)

# Correlation matrix and heatmap
st.subheader("Correlation Matrix and Heatmap")
numeric_cols = data.select_dtypes(include=np.number).columns
corr_matrix = data[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Machine learning model comparison
st.subheader("Machine Learning Model Comparison")
st.markdown("Select two models and feature sets to compare their performance in predicting the happiness score.")

# Defining available models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

# Feature selection options (excluding non-predictive columns)
features = [
    "GDP per capita",
    "Social support",
    "Healthy life expectancy",
    "Freedom to make life choices",
    "Generosity",
    "Perceptions of corruption"
]

# Creating two columns for model comparison
col1, col2 = st.columns(2)

# Model and feature selection for Model 1
with col1:
    st.markdown("### Model 1")
    model1_name = st.selectbox("Select Model 1", list(models.keys()), key="model1")
    selected_features1 = st.multiselect("Select Features for Model 1", features, default=features, key="features1")

# Model and feature selection for Model 2
with col2:
    st.markdown("### Model 2")
    model2_name = st.selectbox("Select Model 2", list(models.keys()), key="model2")
    selected_features2 = st.multiselect("Select Features for Model 2", features, default=features, key="features2")

# Function to train and evaluate model
def train_and_evaluate_model(model, features, data):
    X = data[features]
    y = data["Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return rmse, mae, r2, y_test, y_pred

# Training and evaluating models if features are selected
if selected_features1 and selected_features2:
    st.subheader("Model Performance Comparison")
    
    # Model 1 evaluation
    rmse1, mae1, r21, y_test1, y_pred1 = train_and_evaluate_model(models[model1_name], selected_features1, data)
    st.markdown(f"#### {model1_name} Performance (Features: {', '.join(selected_features1)})")
    st.write(f"RMSE: {rmse1:.4f}")
    st.write(f"MAE: {mae1:.4f}")
    st.write(f"R²: {r21:.4f}")
    
    # Model 2 evaluation
    rmse2, mae2, r22, y_test2, y_pred2 = train_and_evaluate_model(models[model2_name], selected_features2, data)
    st.markdown(f"#### {model2_name} Performance (Features: {', '.join(selected_features2)})")
    st.write(f"RMSE: {rmse2:.4f}")
    st.write(f"MAE: {mae2:.4f}")
    st.write(f"R²: {r22:.4f}")
    
    # Visualizing predictions
    st.subheader("Prediction vs Actual Values")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Model 1 scatter plot
    ax1.scatter(y_test1, y_pred1, alpha=0.5)
    ax1.plot([y_test1.min(), y_test1.max()], [y_test1.min(), y_test1.max()], 'r--', lw=2)
    ax1.set_title(f"{model1_name} Predictions")
    ax1.set_xlabel("Actual Score")
    ax1.set_ylabel("Predicted Score")
    
    # Model 2 scatter plot
    ax2.scatter(y_test2, y_pred2, alpha=0.5)
    ax2.plot([y_test2.min(), y_test2.max()], [y_test2.min(), y_test2.max()], 'r--', lw=2)
    ax2.set_title(f"{model2_name} Predictions")
    ax2.set_xlabel("Actual Score")
    ax2.set_ylabel("Predicted Score")
    
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning("Please select at least one feature for each model to compare performance.")
