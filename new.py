import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

st.set_page_config(layout="wide")

@st.cache_data

def load_data():
    df = pd.read_csv('2019.csv')
    df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
    return df

df = load_data()

st.title("🌍 World Happiness Report 2019: Модели предсказания")
st.markdown("Исследование факторов, влияющих на индекс счастья по странам.")

# Разведочный анализ
st.header("Топ-10 стран по уровню счастья")
top_10 = df.sort_values('score', ascending=False).head(10)
st.dataframe(top_10[['country_or_region', 'score']])

fig1, ax1 = plt.subplots()
sns.barplot(x='score', y='country_or_region', data=top_10, ax=ax1)
st.pyplot(fig1)

st.header("Корреляция между числовыми признаками")
numeric_df = df.select_dtypes(include='number')
fig2, ax2 = plt.subplots(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

st.header("Зависимость счастья от дохода")
fig3, ax3 = plt.subplots()
sns.scatterplot(x='gdp_per_capita', y='score', hue='country_or_region', data=df, ax=ax3)
ax3.legend([],[], frameon=False)
st.pyplot(fig3)

# Подготовка данных
features = ['gdp_per_capita', 'social_support', 'healthy_life_expectancy',
            'freedom_to_make_life_choices', 'generosity', 'perceptions_of_corruption']
X = df[features]
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение моделей и метрики
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

results = []
figs = []
preds_dict = {}
importances_dict = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds_dict[name] = preds

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R²': r2})

    # График
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=preds, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    ax.set_xlabel("Настоящий Score")
    ax.set_ylabel("Предсказанный Score")
    ax.set_title(f"{name}: Предсказание vs Истинное значение")
    ax.grid(True)
    figs.append((name, fig))

    # Важность признаков
    if name in ['Random Forest', 'XGBoost']:
        importances_dict[name] = pd.Series(model.feature_importances_, index=features).sort_values()

# Метрики моделей
st.header("Сравнение моделей")
results_df = pd.DataFrame(results)
st.dataframe(results_df.sort_values(by='RMSE'))

# Визуализация предсказаний
st.header("Визуализация предсказаний")
for name, fig in figs:
    st.subheader(name)
    st.pyplot(fig)

# Важность признаков
st.header("Важность признаков")
if importances_dict:
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    importances_dict['Random Forest'].plot(kind='barh', ax=ax[0], title="Random Forest")
    importances_dict['XGBoost'].plot(kind='barh', ax=ax[1], title="XGBoost")
    st.pyplot(fig)

st.markdown("---")
st.markdown("Проект: [Kaggle Kernel](https://www.kaggle.com/code/glebkabachevskiy/world-happiness-report-2019)")