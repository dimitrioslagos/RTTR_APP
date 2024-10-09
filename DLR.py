import pandas as pd
from matplotlib import pyplot as pl
import openmeteo_requests
import requests_cache
from retry_requests import retry
import configparser
import ast
import math
from RTTR_calculations import compute_RTTR, temperature_calculation
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
import joblib
import os
import plotly.graph_objs as go
import plotly.io as pio
import plotly as pl
from sklearn.model_selection import train_test_split


def calculate_bearing(lat1, lon1, lat2, lon2):
	"""
    Calculate the bearing between two points on the earth.
    The formula assumes that the earth is a perfect sphere.
    """
	# Convert latitudes and longitudes from degrees to radians
	lat1 = math.radians(lat1)
	lon1 = math.radians(lon1)
	lat2 = math.radians(lat2)
	lon2 = math.radians(lon2)

	# Calculate difference in longitudes
	d_lon = lon2 - lon1

	# Calculate bearing using the formula
	x = math.sin(d_lon) * math.cos(lat2)
	y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(d_lon))
	bearing = math.atan2(x, y)

	# Convert bearing from radians to degrees
	bearing_deg = (math.degrees(bearing) + 360) % 360

	return bearing_deg


def calculate_angle_of_attack(lat1, lon1, lat2, lon2, wind_direction):
	"""
    Calculate the angle of attack of wind relative to the line between two points.
    """
	# Calculate the bearing of the line
	line_bearing = calculate_bearing(lat1, lon1, lat2, lon2)

	# Calculate the angle of attack
	angle_of_attack = (wind_direction - line_bearing).abs()

	# Adjust if angle is greater than 180 degrees
	angle_of_attack[angle_of_attack > 180] = 360 - angle_of_attack


	return angle_of_attack


def read_config(filename='settings.cfg'):
    config = configparser.ConfigParser()
    config.read(filename)
    settings = {}
    for section in config.sections():
        for option in config.options(section):
            settings[option] = config.get(section, option)
    return settings

def bring_training_data_ids(I):
	settings = read_config(filename='RTTR_settings.cfg')
	history = ast.literal_eval(settings['history_horizon'])
	ids = I['Mean'].rolling(window=history+1, min_periods=history+1).apply(lambda x: x.notna().all()).dropna()
	return ids.index


def get_historical_data_current(filename):

	DATA = pd.read_csv(filepath_or_buffer=filename,delimiter=',')
	print(DATA.head())
	Tags = DATA.element.unique()
	Measurements = DATA.measure.unique()
	print(Tags)
	Tag1 = Tags[0]
	Tag2 = Tags[2]
	Tag3 = Tags[1]
	Tag4 = Tags[3]

	Current_R = DATA[DATA.measure == ('Current (phase R 40kV)')]
	Current_R1 = Current_R[Current_R.element == Tag1]
	Current_R1.index = pd.DatetimeIndex(Current_R1.timestamp)
	Current_R1 = Current_R1.sort_index()
	Current_R1.drop(columns=['timestamp', 'element', 'measure', 'unit'], inplace=True)
	Current_R1.index = Current_R1.index.round('T')
	Current_R1 = Current_R1[~Current_R1.index.duplicated(keep='first')]

	Current_R2 = Current_R[Current_R.element == Tag3]
	Current_R2.index = pd.DatetimeIndex(Current_R2.timestamp)
	Current_R2 = Current_R2.sort_index()
	Current_R2.drop(columns=['timestamp', 'element', 'measure', 'unit'], inplace=True)
	Current_R2.index = Current_R2.index.round('T')
	Current_R2 = Current_R2[~Current_R2.index.duplicated(keep='first')]

	Current_S = DATA[DATA.measure == ('Current (phase S 40kV)')]
	Current_S1 = Current_S[Current_S.element == Tag1]
	Current_S1.index = pd.DatetimeIndex(Current_S1.timestamp)
	Current_S1 = Current_S1.sort_index()
	Current_S1.drop(columns=['timestamp', 'element', 'measure', 'unit'], inplace=True)
	Current_S1.index = Current_S1.index.round('T')
	Current_S1 = Current_S1[~Current_S1.index.duplicated(keep='first')]

	Current_S2 = Current_S[Current_S.element == Tag3]
	Current_S2.index = pd.DatetimeIndex(Current_S2.timestamp)
	Current_S2 = Current_S2.sort_index()
	Current_S2.drop(columns=['timestamp', 'element', 'measure', 'unit'], inplace=True)
	Current_S2.index = Current_S2.index.round('T')
	Current_S2.index = Current_S2.index.round('T')
	Current_S2 = Current_S2[~Current_S2.index.duplicated(keep='first')]

	Current_T = DATA[DATA.measure == ('Current (phase T 40kV)')]
	Current_T1 = Current_T[Current_T.element == Tag1]
	Current_T1.index = pd.DatetimeIndex(Current_T1.timestamp)
	Current_T1 = Current_T1.sort_index()
	Current_T1.drop(columns=['timestamp', 'element', 'measure', 'unit'], inplace=True)
	Current_T1.index = Current_T1.index.round('T')
	Current_T1 = Current_T1[~Current_T1.index.duplicated(keep='first')]

	Current_T2 = Current_T[Current_T.element == Tag3]
	Current_T2.index = pd.DatetimeIndex(Current_T2.timestamp)
	Current_T2 = Current_T2.sort_index()
	Current_T2.drop(columns=['timestamp', 'element', 'measure', 'unit'], inplace=True)
	Current_T2.index = Current_T2.index.round('T')
	Current_T2 = Current_T2[~Current_T2.index.duplicated(keep='first')]

	Currenta = pd.concat((Current_T1, Current_S1, Current_R1), axis=1)
	Currenta.columns = ['T1', 'S1', 'R1']
	Currenta['Mean'] = (Currenta['T1'] + Currenta['S1'] + Currenta['R1']) / 3

	Currentb = pd.concat((Current_T2, Current_S2, Current_R2), axis=1)
	Currentb.columns = ['T2', 'S2', 'R2']
	Currentb['Mean2'] = (Currentb['T2'] + Currentb['S2'] + Currentb['R2']) / 3

	Currents = Currenta
	Currents.drop(index=Currents.index[Currents.Mean == 0], inplace=True)
	I = Currents.resample('H').mean().dropna(axis=0)
	return I

def get_weather_data(ids, lat,lon):
	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	# Make sure all required weather variables are listed here
	# The order of variables in hourly or daily is important to assign them correctly below
	url = "https://archive-api.open-meteo.com/v1/archive"
	params = {
		"latitude": lat,
		"longitude": lon,
		"start_date": str(ids[0].year)+"-01-01",
		"end_date": str(ids[0].year)+"-12-31",
		"hourly": ["temperature_2m",'wind_speed_10m','wind_direction_10m'],
		"wind_speed_unit":'ms'
	}
	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]

	# Process hourly data. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
	hourly_ws = hourly.Variables(1).ValuesAsNumpy()
	hourly_wd = hourly.Variables(2).ValuesAsNumpy()

	hourly_data = {"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}
	hourly_data["temperature_2m"] = hourly_temperature_2m
	hourly_data["wind_speed"] = hourly_ws
	hourly_data["wind_direction"] = hourly_wd

	hourly_dataframe = pd.DataFrame(data = hourly_data)
	hourly_dataframe['date'] = hourly_dataframe['date'] + pd.Timedelta(hours=1)
	hourly_dataframe.index = hourly_dataframe['date']
	hourly_dataframe.index  =hourly_dataframe.index.tz_localize(None)
	Weather = hourly_dataframe.drop(columns=['date'])
	#Weather = Weather.resample('15T').asfreq().interpolate()
	return Weather.loc[ids]

def read_static_data(filename):
    Lines = pd.read_excel(filename,sheet_name='Feeder_data')
    return Lines

def prepare_training_data(I,ids):
	settings = read_config(filename='RTTR_settings.cfg')
	history = ast.literal_eval(settings['history_horizon'])
	prediction_horizon = ast.literal_eval(settings['prediction_horizon'])
	columns = ['now']+[str(x)+'_hours_ago' for x in range(1,history+1)]
	Train_X = pd.DataFrame(index=ids,columns=columns)
	Imax = I.max(axis=1)
	Train_X['moving_avg_3h'] = Imax.rolling(window=3).mean()
	Train_X['moving_avg_6h'] = Imax.rolling(window=6).mean()
	Train_X.loc[ids,'now'] = Imax.loc[ids]
	Train_X.loc[ids, 'hour'] = Imax.loc[ids].index.hour
	Train_X.loc[ids, 'month'] = Imax.loc[ids].index.month
	for h in range(1,history+1):
		Train_X.loc[:,str(h)+'_hours_ago'] = Imax.shift(h).loc[ids]
	##Prepare predictions
	Train_X['change_from_3h_ago'] = Train_X['now'] - Train_X['3_hours_ago']
	Train_X['change_from_6h_ago'] = Train_X['now'] - Train_X['6_hours_ago']
	columns_y =  [str(x) + '_hours_ahead' for x in range(1, prediction_horizon + 1)]
	Train_Y = pd.DataFrame(index=ids, columns=columns_y)
	for h in range(1,prediction_horizon+1):
		Train_Y.loc[:,str(h)+'_hours_ahead'] = Imax.shift(-h).loc[ids]
	return Train_X, Train_Y

def train_models(X,Y,h):
	settings = read_config(filename='RTTR_settings.cfg')
	prediction_horizon = ast.literal_eval(settings['prediction_horizon'])
	#find rows with nans and remove them
	Yi = Y[str(h)+'_hours_ahead']
	Yi.dropna(inplace=True)
	Xi = X.loc[Yi.index,:]
	# Split data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(Xi, Yi, test_size=0.2, random_state=42)

	##Random Forest
	qrf = RandomForestQuantileRegressor(min_samples_leaf=5,n_jobs=2,default_quantiles=[0.1,0.5,0.95])
	qrf.fit(X_train, y_train)
	y_pred = qrf.predict(X_test, quantiles=[0.1, 0.5, 0.95])

	Results = pd.DataFrame(columns=['actual','maxV','minV','meanV'])
	Results.actual = y_test.values
	Results.meanV = y_pred[:,1]
	Results.maxV = y_pred[:,2]
	Results.minV = y_pred[:,0]
	Results.index = y_test.index
	# Assume `rf` is your trained RandomForestRegressor model
	return qrf

def prepare_attributes_to_forecast(I,t):
	settings = read_config(filename='RTTR_settings.cfg')
	history = ast.literal_eval(settings['history_horizon'])
	columns = ['now']+[str(x)+'_hours_ago' for x in range(1,history+1)]
	X = pd.DataFrame(index=[t],columns=columns)
	Imax = I.max(axis=1)
	X['moving_avg_3h'] = (Imax.rolling(window=3).mean()).loc[t]
	X['moving_avg_6h'] = (Imax.rolling(window=6).mean()).loc[t]
	X.loc[t,'now'] = Imax.loc[t]
	X.loc[t, 'hour'] = t.hour
	X.loc[t, 'month'] = t.month
	for h in range(1,history+1):
		X.loc[t,str(h)+'_hours_ago'] = Imax.shift(h).loc[t]
	##Prepare predictions
	X['change_from_3h_ago'] = X['now'] - X['3_hours_ago']
	X['change_from_6h_ago'] = X['now'] - X['6_hours_ago']
	return X


def get_prediction_models(filename,filename2):
	# Define the directory and file path
	directory = "models"
	models = []
	# Check if the directory exists
	if not os.path.exists(directory):
		# If the directory does not exist, create it
		os.makedirs(directory)
		print(f"Directory '{directory}' created.")
	##Get settings
	settings = read_config(filename='RTTR_settings.cfg')
	prediction_horizon = ast.literal_eval(settings['prediction_horizon'])
	I = get_historical_data_current(filename2)
	for h in range(1,prediction_horizon+1):
		model_filename = str(h)+"_hours_random_forest_model.pkl"
		model_path = os.path.join(directory, model_filename)
		# Check if the model file exists
		if os.path.isfile(model_path):
			print(f"Model file '{model_filename}' exists in '{directory}'. Loading the model...")
			# Load the model
			models.append(joblib.load(model_path))
			print("Model loaded successfully.")
		else:
			print(f"Model file '{model_filename}' does not exist in '{directory}'. Training and saving the model...")

			ids = bring_training_data_ids(I)
			X, Y = prepare_training_data(I, ids)
			qrf = train_models(X, Y,h)
			models.append(qrf)
			joblib.dump(qrf, model_path)
	return models


def compute_predictions(models,I,t):
	X = prepare_attributes_to_forecast(I, t)
	predictions = pd.DataFrame(index=range(1,len(models)+1),columns=['0.1','0.5','0.9'])
	for hour in range(1,len(models)+1):
		predictions.loc[hour,:] = models[hour-1].predict(X, quantiles=[0.1, 0.5, 0.95])
	return predictions

def get_loadings(preds):
	Lines = read_static_data('Static_data.xlsx')
	Lines_OH = Lines[Lines['OverHead_or_UnderGround'] == 'OH']
	Loading_10 = pd.DataFrame(index=I.index[1435:1435 + len(models)], columns=Lines_OH.index)
	Loading_50 = pd.DataFrame(index=I.index[1435:1435 + len(models)], columns=Lines_OH.index)
	Loading_90 = pd.DataFrame(index=I.index[1435:1435 + len(models)], columns=Lines_OH.index)
	Loading_static = pd.DataFrame(index=I.index[1435:1435 + len(models)], columns=Lines_OH.index)

	for it, line in Lines_OH.iterrows():
		weather_data = get_weather_data(Loading_10.index,
										lat=(line.lat_start + line.lat_end) / 2,
										lon=(line.lon_start + line.lon_end) / 2)
		Angle_of_attack = calculate_angle_of_attack(line.lat_start, line.lon_start, line.lat_end,
													line.lon_end, weather_data['wind_direction'])
		weather_data['wind_direction'] = Angle_of_attack

		t_range = weather_data.index

		Rac = pd.DataFrame(
			[line['resistance (ohm/km)'] / 1000, (1 + 0.004 * 60) * (line['resistance (ohm/km)']) / 1000],
			index=[25, 85])
		diameter = (np.sqrt(line['section (m^2)'] / np.pi) * 2) / 1000
		Ic = compute_RTTR(ts=85, wind_speed=weather_data['wind_speed'], wind_angle=weather_data['wind_direction'],
						  temperature=weather_data['temperature_2m'], t_range=t_range,
						  latitude=(line.lat_start + line.lat_end) / 2, Rac=Rac,
						  diameter=diameter)
		preds.index = t_range
		Ti50 = temperature_calculation(I=preds['0.5'],lat=(line.lat_start + line.lat_end) / 2,
								t_range=t_range,Rac=Rac,diameter=diameter,
								wind_angle=weather_data['wind_direction'],wind_speed=weather_data['wind_speed'],
								temperature=weather_data['temperature_2m'])
		Ti90 = temperature_calculation(I=preds['0.9'],lat=(line.lat_start + line.lat_end) / 2,
								t_range=t_range,Rac=Rac,diameter=diameter,
								wind_angle=weather_data['wind_direction'],wind_speed=weather_data['wind_speed'],
								temperature=weather_data['temperature_2m'])
		Ti10 = temperature_calculation(I=preds['0.1'],lat=(line.lat_start + line.lat_end) / 2,
								t_range=t_range,Rac=Rac,diameter=diameter,
								wind_angle=weather_data['wind_direction'],wind_speed=weather_data['wind_speed'],
								temperature=weather_data['temperature_2m'])

		Loading_10.loc[:, it] = 100 * preds.loc[:, '0.1'].values / Ic
		Loading_50.loc[:, it] = 100 * preds.loc[:, '0.5'].values / Ic
		Loading_90.loc[:, it] = 100 * preds.loc[:, '0.9'].values / Ic
		Loading_static.loc[:, it] = 100 * preds.loc[:, '0.5'].values / line['current_admissable (A)']
	return Loading_static, Loading_10, Loading_50, Loading_90, Ti10, Ti50, Ti90

def plot_loadings(Ls,L10,L50,L90):

	layout = go.Layout(
		legend=dict(
			orientation='h',  # Horizontal orientation
			yanchor='bottom',  # Anchoring to the bottom of the legend
			y=1.0,  # Position the legend above the plot area
			xanchor='center',  # Center the legend horizontally
			x=0.8 # Center the legend horizontally
		),
		xaxis=dict(title='Time'),
		yaxis=dict(title='Loading (%)',
				   tickmode='array',
				   tickvals=[value for value in range(0,int(np.round(Ls.max().max()/10)*10+10),5)],  # Define which y-axis ticks to show
				   ticktext=[str(value) for value in range(0,int(np.round(Ls.max().max()/10)*10+10),5)],))

	fig = go.Figure(layout=layout)

	fig.add_traces(go.Scatter(
		x=L10.index,
		y=L10.max(axis=1),
		line=dict(color='gray'),
		name='DLR 10%',
		showlegend=False
	))
	fig.add_traces(go.Scatter(
		x=L90.index,
		y=L90.max(axis=1),
		fill='tonexty',  # Fill area between the previous trace (col1) and this trace
		line=dict(color='gray'),
		name='DLR 90%',
		showlegend=False
	))


	fig.add_traces(go.Scatter(
		x=L50.index,
		y=L50.max(axis=1),
		line=dict(color='green'),
		name='DLR 50%',
		showlegend=True
	))

	fig.add_traces(go.Scatter(
		x=Ls.index,
		y=Ls.max(axis=1),
		line=dict(color='red'),
		name='Static Rating',
		showlegend=True
	))

	# Save the figure as an HTML file
	pio.write_html(fig, file='loading_plot.html', auto_open=True)

	return 0


def plot_temperatures(Τ10,Τ50,Τ90):

	layout = go.Layout(
		legend=dict(
			orientation='h',  # Horizontal orientation
			yanchor='bottom',  # Anchoring to the bottom of the legend
			y=1.0,  # Position the legend above the plot area
			xanchor='center',  # Center the legend horizontally
			x=0.8  # Center the legend horizontally
		),
		xaxis=dict(title='Time'),
		yaxis=dict(title='Temperature (C)',
				   tickmode='array',
				   tickvals=[value for value in range(0,90,5)],  # Define which y-axis ticks to show
				   ticktext=[str(value) for value in range(0,90,5)],))

	fig = go.Figure(layout=layout)

	fig.add_traces(go.Scatter(
		x=T10.index,
		y=T10.max(axis=1),
		line=dict(color='gray'),
		name='10%',
		showlegend=False
	))
	fig.add_traces(go.Scatter(
		x=T90.index,
		y=T90.max(axis=1),
		fill='tonexty',  # Fill area between the previous trace (col1) and this trace
		line=dict(color='gray'),
		name='90%',
		showlegend=False
	))


	fig.add_traces(go.Scatter(
		x=T50.index,
		y=T50.max(axis=1),
		line=dict(color='green'),
		name='50%',
		showlegend=True
	))

	fig.add_traces(go.Scatter(
		x=T10.index,
		y=[85]*T10.shape[0],
		line=dict(color='red'),
		name='Temperature Rating',
		showlegend=True
	))

	# Save the figure as an HTML file
	pio.write_html(fig, file='temperature_plot.html', auto_open=True)

	return 0
# Lines = read_static_data('Static_data.xlsx')
# I = get_historical_data_current('Historical_Data.csv')
# #
# models = get_prediction_models('Static_data.xlsx','Historical_Data.csv')
# preds = compute_predictions(models,I,I.index[1435])
# Ls,L10,L50,L90,T10, T50, T90=get_loadings(preds)
# #
# plot_loadings(Ls,L10,L50,L90)
# plot_temperatures(T10,T50,T90)
