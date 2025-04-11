import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor, StackingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

df = pd.read_csv("ship_fuel_efficiency.csv")
X = df[["ship_type", "distance"]]
y = df["fuel_consumption"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_models = [
    ("rf", RandomForestRegressor(random_state=42)),
    ("gb", GradientBoostingRegressor(random_state=42)),
    ("ada", AdaBoostRegressor(random_state=42)),
    ("xgb", XGBRegressor(random_state=42))
]

final_model = GradientBoostingRegressor(random_state=42)

stacking_reg = StackingRegressor(estimators=base_models, final_estimator=final_model)

stacking_reg.fit(X_train, y_train)
stack_preds = stacking_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, stack_preds))
r2 = r2_score(y_test, stack_preds)


print(f"  RMSE     : {rmse:.2f}")
print(f"  R² Score : {r2:.4f}")
