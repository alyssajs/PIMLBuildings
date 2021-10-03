import matplotlib.pyplot as plot
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from datetime import datetime

np.random.seed(5)
def normalization(data_df, y_col):
    min_val = data_df[y_col].min()
    max_val = data_df[y_col].max()
    mean_val = data_df[y_col].mean()
    data_df[y_col] = (data_df[y_col]- mean_val) / (max_val - min_val)
    return [data_df, min_val, mean_val, max_val]

#goal: predict room temperature, air flow rate, and reheat?

samples_15_from_30 = pd.read_csv('C:\\Users\\rubbi\\PycharmProjects\\pimlBuildings\\B90_102\\experiment_30m\\HVAC_B90_102_exp_30m_20210426_15mins.csv',
                                 parse_dates=['time'])


subset_samples = samples_15_from_30[['room_temp_smooth',
                                     'htg_valve_position',
                                     'airflow_current_smooth',
                                     'supply_discharge_temp_smooth',
                                     'thermostat_outside_temp_smooth',
                                     'setpoint',
                                     'htg_sp_current',
                                     'airflow_desired',
                                     'occupied_status']]

# = len(subset_samples['time'])
#for i in range(1, numRow):
#    subset_samples['time'][i] = subset_samples['time'][i].timestamp()


vals = subset_samples.values
min_max_scaler = preprocessing.MinMaxScaler()
vals_scaled = min_max_scaler.fit_transform(vals)
normalized_data = pd.DataFrame(vals_scaled, columns=subset_samples.columns)
print(normalized_data.head())
'''
#to predict room_temp - potential predictors are
#time
#supply_discharge_temp
#thermostat_outside_temp
#htg_sp_current
#setpoint
#htg_valve_position_ave
#htg_signal
#clg_signal



#visualize relationships
print(samples_15_from_30.columns)
sns.pairplot(samples_15_from_30, kind="scatter", x_vars=["supply_discharge_temp_smooth",
                                                         "thermostat_outside_temp_smooth"],
             y_vars=["room_temp", "htg_valve_position", "airflow_current"])
sns.pairplot(samples_15_from_30, kind="scatter",x_vars=["setpoint",
                                                        "htg_sp_current",
                                                        "setpoint"],
            y_vars=["room_temp", "htg_valve_position", "airflow_current"])
sns.pairplot(samples_15_from_30, kind="scatter",x_vars=["time"],
            y_vars=["room_temp", "htg_valve_position", "airflow_current"])
plot.show()

#normalize data

'''