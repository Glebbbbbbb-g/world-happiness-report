import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

df = pd.read_csv('2019.csv')
print(df.head())
print(df.describe())
df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]

top_10 = df.sort_values('score', ascending=False).head(10)
print(top_10)

sns.barplot(x='score', y='country_or_region', data=top_10)
plt.show()

numeric_df = df.select_dtypes(include='number')
plt.figure(figsize = (10,7))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.show()

sns.scatterplot(x='gdp_per_capita', y='score', hue='country_or_region', data=df)
plt.show()

features = ['gdp_per_capita', 'social_support', 'healthy_life_expectancy',
            'freedom_to_make_life_choices', 'generosity', 'perceptions_of_corruption']
X = df[features]
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = mean_squared_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)

print("Linear Regression:")
print("MAE:", lr_mae)
print("RMSE:", lr_rmse)
print("R²:", lr_r2)

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=lr_preds, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Настоящий Score")
plt.ylabel("Предсказанный Score")
plt.title("Linear Regression: Предсказание vs Истинное значение")
plt.grid(True)
plt.show()


rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Метрики
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = mean_squared_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

print("Random Forest:")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2:", rf_r2)

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=rf_preds, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Настоящий Score")
plt.ylabel("Предсказанный Score")
plt.title("Random Forest: Предсказание vs Истинное значение")
plt.grid(True)
plt.show()


xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

# Метрики
xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_rmse = mean_squared_error(y_test, xgb_preds)
xgb_r2 = r2_score(y_test, xgb_preds)

print("XGBoost:")
print("MAE:", xgb_mae)
print("RMSE:", xgb_rmse)
print("R2:", xgb_r2)


plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=xgb_preds, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Настоящий Score")
plt.ylabel("Предсказанный Score")
plt.title("XGBoost: Предсказание vs Истинное значение")
plt.grid(True)
plt.show()


results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'MAE': [lr_mae, rf_mae, xgb_mae],
    'RMSE': [lr_rmse, rf_rmse, xgb_rmse],
    'R²': [lr_r2, rf_r2, xgb_r2]
})

results.sort_values(by='RMSE')
print(results)

rf_importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
xgb_importances = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=True)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
rf_importances.plot(kind='barh', ax=ax[0], title="Random Forest")
xgb_importances.plot(kind='barh', ax=ax[1], title="XGBoost")
plt.tight_layout()
plt.show()