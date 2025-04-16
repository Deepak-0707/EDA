import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


df = pd.read_csv("ship_fuel_efficiency.csv")

X = df[["ship_type", "distance"]]
y = df["fuel_consumption"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "Extra Trees": ExtraTreesRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "XGBoost":XGBRegressor(random_state=42)
}


model_preds = {}
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append((name, rmse, r2))
    model_preds[name] = preds

    print(f"\n{name}")
    print(f"   RMSE     : {rmse:.2f}")
    print(f"   R² Score : {r2:.4f}")

best_model = min(results, key=lambda x: x[1])
print(f"\nBest Model Based on RMSE: {best_model[0]} (RMSE: {best_model[1]:.2f}, R²: {best_model[2]:.4f})")

for model_name, preds in model_preds.items():
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=preds, color="teal", edgecolor="white", s=60)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
    plt.xlabel("Actual Fuel Consumption")
    plt.ylabel("Predicted Fuel Consumption")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
