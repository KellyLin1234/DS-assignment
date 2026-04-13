!pip install pandas numpy scikit-learn xgboost
import pandas as pd

file_path = r"C:\Users\Kitty Lim\Downloads\Gold Price.csv"
df = pd.read_csv(file_path)

df.head()
import numpy as np

for col in ['Price','Open','High','Low']:
    df[col] = df[col].astype(str).str.replace(',','', regex = False)
    df[col] = df[col].astype(float)
    # as to remove "," change to float

def convert_volume(v):
    v = str(v)
    if 'K' in v:
        return float (v.replace('K',''))* 1e3
    elif 'M' in v:
        return float (v.replace('M',''))* 1e6
    else:
        return float(v)

df['Volume'] = df['Volume'].apply(convert_volume)
    # as to process volume (K=thousand, M= million) 

df['Chg%'] = df['Chg%'].astype(str).str.replace('%','', regex = False)
df['Chg%'] = df['Chg%'].astype(float)
    # as to process Chg%

df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].map(pd.Timestamp.toordinal)
    # as to process Date to Time then change to number

df.head
X = df[['Date','Open','High','Low','Volume','Chg%']]

y = df['Price']

from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size = 0.2, random_state = 42)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score (y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true))* 100

    return rmse, mae, r2, mape
from sklearn.ensemble import GradientBoostingRegressor

print ("- - - - Gradient Boosting - - - -")

gb_default = GradientBoostingRegressor(random_state = 42)
gb_default.fit(X_train, y_train)
y_pred_gb_default = gb_default.predict(X_test)

rmse_gb_default = np.sqrt(mean_squared_error(y_test, y_pred_gb_default))
mae_gb_default = mean_absolute_error(y_test, y_pred_gb_default)
r2_gb_default = r2_score(y_test, y_pred_gb_default)
mape_gb_default = mean_absolute_percentage_error(y_test, y_pred_gb_default)
print(f"Default GB -> RMSE: {rmse_gb_default:.2f}, MAE: {mae_gb_default:.2f}, R²: {r2_gb_default:.4f}, MAPE: {mape_gb_default:.2f}%")

gb_manual = GradientBoostingRegressor(n_estimators=150, learning_rate=0.5, max_depth=4, random_state=42)
gb_manual.fit(X_train, y_train)
y_pred_gb_manual = gb_manual.predict(X_test)

rmse_gb_manual = np.sqrt(mean_squared_error(y_test, y_pred_gb_manual))
mae_gb_manual = mean_absolute_error(y_test, y_pred_gb_manual)
r2_gb_manual = r2_score(y_test, y_pred_gb_manual)
mape_gb_manual = mean_absolute_percentage_error(y_test, y_pred_gb_manual)
print(f"Manual Tuned GB -> RMSE: {rmse_gb_manual:.2f}, MAE: {mae_gb_manual:.2f}, R²: {r2_gb_manual:.4f}, MAPE: {mape_gb_manual:.2f}%")

param_grid_gb = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}

grid_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, cv=3, scoring='neg_mean_squared_error')
grid_gb.fit(X_train, y_train)

best_gb = grid_gb.best_estimator_
y_pred_gb_best = best_gb.predict(X_test)

rmse_gb_best = np.sqrt(mean_squared_error(y_test, y_pred_gb_best))
mae_gb_best = mean_absolute_error(y_test, y_pred_gb_best)
r2_gb_best = r2_score(y_test, y_pred_gb_best)
mape_gb_best = mean_absolute_percentage_error(y_test, y_pred_gb_best)

print(f"Best GB Params: {grid_gb.best_params_}")
print(f"Best GB -> RMSE: {rmse_gb_best:.2f}, MAE: {mae_gb_best:.2f}, R²: {r2_gb_best:.4f}, MAPE: {mape_gb_best:.2f}%")


from xgboost import XGBRegressor

print("\n----- XGBoost -----")

xgb_default = XGBRegressor(random_state=42, objective='reg:squarederror')
xgb_default.fit(X_train, y_train)
y_pred_xgb_default = xgb_default.predict(X_test)

rmse_xgb_default = np.sqrt(mean_squared_error(y_test, y_pred_xgb_default))
mae_xgb_default = mean_absolute_error(y_test, y_pred_xgb_default)
r2_xgb_default = r2_score(y_test, y_pred_xgb_default)
mape_xgb_default = mean_absolute_percentage_error(y_test, y_pred_xgb_default)
print(f"Default XGB -> RMSE: {rmse_xgb_default:.2f}, MAE: {mae_xgb_default:.2f}, R²: {r2_xgb_default:.4f}, MAPE: {mape_xgb_default:.2f}%")

xgb_manual = XGBRegressor(n_estimators=150, learning_rate=0.5, max_depth=4, random_state=42)
xgb_manual.fit(X_train, y_train)
y_pred_xgb_manual = xgb_manual.predict(X_test)

rmse_xgb_manual = np.sqrt(mean_squared_error(y_test, y_pred_xgb_manual))
mae_xgb_manual = mean_absolute_error(y_test, y_pred_xgb_manual)
r2_xgb_manual = r2_score(y_test, y_pred_xgb_manual)
mape_xgb_manual = mean_absolute_percentage_error(y_test, y_pred_xgb_manual)
print(f"Manual Tuned XGB -> RMSE: {rmse_xgb_manual:.2f}, MAE: {mae_xgb_manual:.2f}, R²: {r2_xgb_manual:.4f}, MAPE: {mape_xgb_manual:.2f}%")

param_grid_xgb = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}

grid_xgb = GridSearchCV(XGBRegressor(random_state=42, objective='reg:squarederror'), param_grid_xgb, cv=3, scoring='neg_mean_squared_error')
grid_xgb.fit(X_train, y_train)

best_xgb = grid_xgb.best_estimator_
y_pred_xgb_best = best_xgb.predict(X_test)

rmse_xgb_best = np.sqrt(mean_squared_error(y_test, y_pred_xgb_best))
mae_xgb_best = mean_absolute_error(y_test, y_pred_xgb_best)
r2_xgb_best = r2_score(y_test, y_pred_xgb_best)
mape_xgb_best = mean_absolute_percentage_error(y_test, y_pred_xgb_best)

print(f"Best XGB Params: {grid_xgb.best_params_}")
print(f"Best XGB -> RMSE: {rmse_xgb_best:.2f}, MAE: {mae_xgb_best:.2f}, R²: {r2_xgb_best:.4f}, MAPE: {mape_xgb_best:.2f}%")
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train,y_train)
y_pred_gbr = gbr.predict(X_test)

from xgboost import XGBRegressor
xgb = XGBRegressor (random_state=42, objective= 'reg:squarederror')
xgb.fit(X_train,y_train)
y_pred_xgb = xgb.predict(X_test)

# Gradient Boosting
rmse_gbr, mae_gbr, r2_gbr, mape_gbr = evaluate (y_test, y_pred_gbr)

print("Gradient Boosting Results:")
print(f"RMSE:{rmse_gbr:.4f}")
print(f"MAE:{mae_gbr:.4f}")
print(f"R2:{r2_gbr:.4f}")
print(f"MAPE:{mape_gbr:.4f}")

# XGBoost
rmse_xgb, mae_xgb, r2_xgb, mape_xgb = evaluate(y_test, y_pred_xgb)

print("\nXGBoost Results:")
print(f"RMSE:{rmse_xgb:.4f}")
print(f"MAE:{mae_xgb:.4f}")
print(f"R2:{r2_xgb:.4f}")
print(f"MAPE:{mape_xgb:.4f}")
import pandas as pd

results = pd.DataFrame({
    'Model': ['XGBoost','Gradient Boosting'],
    'RMSE': [rmse_xgb,rmse_gbr],
    'MAE': [mae_xgb,mae_gbr],
    'R2' : [r2_xgb,r2_gbr],
    'MAPE': [mape_xgb, mape_xgb]
})

results.round(4)
import pandas as pd

results = pd.DataFrame({
    'Model': ['GB Default', 'GB Manual', 'XGB Default', 'XGB Manual'],
    'RMSE': [rmse_gb_default, rmse_gb_manual, rmse_xgb_default, rmse_xgb_manual],
    'MAE': [mae_gb_default, mae_gb_manual, mae_xgb_default, mae_xgb_manual],
    'R2': [r2_gb_default, r2_gb_manual, r2_xgb_default, r2_xgb_manual],
    'MAPE': [mape_gb_default, mape_gb_manual, mape_xgb_default, mape_xgb_manual]
})

print(results)

