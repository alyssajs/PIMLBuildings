import matplotlib.pyplot as plot
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime

np.random.seed(5)
def normalization(data_df, y_col):
    min_val = data_df[y_col].min()
    max_val = data_df[y_col].max()
    mean_val = data_df[y_col].mean()
    data_df[y_col] = (data_df[y_col]- mean_val) / (max_val - min_val)
    return [data_df, min_val, mean_val, max_val]


samples_15_from_30 = pd.read_csv('C:\\Users\\rubbi\\PycharmProjects\\pimlBuildings\\B90_102\\experiment_30m\\HVAC_B90_102_exp_30m_20210426_15mins.csv',
                                 parse_dates=['time'])




features = samples_15_from_30[['ahu_supply_temp', 'setpoint']]
outputs = samples_15_from_30[['room_temp_smooth', 'htg_valve_position', 'airflow_current']]

feature_vals =features.values
output_vals = outputs.values
min_max_scaler = preprocessing.MinMaxScaler()
feature_vals_scaled = min_max_scaler.fit_transform(feature_vals)
output_vals_scaled = min_max_scaler.fit_transform(output_vals)
normalized_features = pd.DataFrame(feature_vals_scaled, columns=features.columns)
normalized_outputs = pd.DataFrame(output_vals_scaled, columns=outputs.columns)
in_train, in_test, out_train, out_test = train_test_split(normalized_features, normalized_outputs, train_size=0.66, random_state=1)


model = DecisionTreeRegressor()
model.fit(in_train, out_train)
print(model.n_features_)
print(model.n_outputs_)
print(model.score(in_test, out_test))
test_pred = model.predict(in_test)


predicted_room_temp = test_pred[:,0]
predicted_htg= test_pred[:,1]
predicted_airflow = test_pred[:,2]
actual_temp = out_test['room_temp_smooth']
actual_htg  = out_test['htg_valve_position']
actual_airflow = out_test['airflow_current']
index = range(0, len(predicted_room_temp))
plot.scatter(actual_temp, predicted_room_temp, color='green')
plot.show()
plot.scatter(actual_htg, predicted_htg, color='red')
plot.show()
plot.scatter(actual_airflow, predicted_airflow, color='blue')
plot.show()
test_mse_temp = np.mean(np.power(np.subtract(predicted_room_temp, actual_temp), 2))
print("Temperature Test MSE: ", test_mse_temp)
test_mse_htg = np.mean(np.power(np.subtract(predicted_htg, actual_htg), 2))
print("Heating Test MSE: ", test_mse_htg)
test_mse_airflow = np.mean(np.power(np.subtract(predicted_airflow, actual_airflow), 2))
print("Airflow Test MSE: ", test_mse_airflow)
