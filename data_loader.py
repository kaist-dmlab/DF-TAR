###############
# Data Loader #
###############

import pandas as pd
import numpy as np
import json

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as mae 
from sklearn.metrics import mean_squared_error as mse


def data_loader(data_path, city, year, level='district', length=12, n_steps=12, is_scale=False):
    
    def normalize(train, test):
        # scaling features to range (0, 1) by MinMaxScaler
        if is_scale:
            scaler = MinMaxScaler()
            train_shape, test_shape = train.shape, test.shape
            train = scaler.fit_transform(train.reshape(-1, train_shape[-1]))
            test = scaler.transform(test.reshape(-1, test_shape[-1]))
            return train.reshape(train_shape), test.reshape(test_shape)
        else:
            return train, test

    risk_data = pd.read_csv(f'{data_path}/risk_scores/{city}-{year}-{level}-hour-risk.csv')
    selected_areas = risk_data.drop(columns=['date', 'time']).columns
    n_districts = len(selected_areas) # number of districts
    n_outputs = len(selected_areas)

    # Traffic Accident Risk
    risk_train, y_train = [], []
    risk_test, y_test = [], []
    for i in range(length, 721-n_steps):
        if i <= (21*24): # before date 22nd
            y_train.append(risk_data.drop(columns=['date', 'time']).iloc[i:i+n_steps, :n_outputs].to_numpy())
            risk_train.append(risk_data.drop(columns=['date', 'time']).iloc[i-length:i, :n_districts].to_numpy())
        else:
            y_test.append(risk_data.drop(columns=['date', 'time']).iloc[i:i+n_steps, :n_outputs].to_numpy())
            risk_test.append(risk_data.drop(columns=['date', 'time']).iloc[i-length:i, :n_districts].to_numpy())
        
    risk_train, risk_test = normalize(np.array(risk_train), np.array(risk_test))
    y_train, y_test = np.array(y_train), np.array(y_test)

    # Weather & Air Quality  
    weather_data = pd.read_csv(f'{data_path}/weather/{city}-{year}-count.csv').fillna(0)
    if level == 'district':
        weather_data['location'] = weather_data['location'].apply(lambda x: x.split('|')[0])
        weather_data = weather_data.groupby(by=['date','time','location'], as_index=False).mean()                
    weather_train, weather_test = [], []

    location_weather = []
    for location in selected_areas:
        location_weather.append(weather_data[weather_data['location'] == location].iloc[:, 3:].to_numpy())

    location_weather = np.concatenate(location_weather, axis=1)

    for i in range(length, 721-n_steps):
        if i <= (21*24):
            weather_train.append(location_weather[i-length:i])
        else:
            weather_test.append(location_weather[i-length:i])
    
    weather_train, weather_test = normalize(np.array(weather_train).reshape(len(weather_train), length, n_districts, -1), np.array(weather_test).reshape(len(weather_test), length, n_districts, -1))


    # Dangerous Driving Behavior
    dtg_data = pd.read_csv(f'{data_path}/dangerous_cases/{city}-{year}-date-hour-{level}-new.csv')
    dtg_train, dtg_test = [], []

    location_dtg = []
    for location in selected_areas:
        if level == 'district':
            district = location.split('|')[0]
            location_dtg.append(dtg_data[dtg_data['district'] == district].iloc[:, 3:].to_numpy())
        else:
            district, subdistrict = location.split('|')[0], location.split('|')[1]
            location_dtg.append(dtg_data[(dtg_data['district'] == district) & (dtg_data['subdistrict'] == subdistrict)].iloc[:, 3:].to_numpy())

    location_dtg = np.concatenate(location_dtg, axis=1)

    for i in range(length, 721-n_steps):
        if i <= (21*24):
            dtg_train.append(location_dtg[i-length:i])
        else:
            dtg_test.append(location_dtg[i-length:i])

    dtg_train, dtg_test = normalize(np.array(dtg_train).reshape(len(dtg_train), length, n_districts, -1), np.array(dtg_test).reshape(len(dtg_test), length, n_districts, -1))


    # Road data
    road_data = pd.read_csv(f'{data_path}/roads/{city}-{year}-{level}-road-count.csv').drop(columns=['attribute'])
    road_train, road_test = [], []

    location_road = []
    for location in selected_areas:
        location_road.append(road_data[location].to_numpy())

    for i in range(length, 721-n_steps):
        if i <= (21*24):
            road_train.append(np.array([location_road]*length))
        else:
            road_test.append(np.array([location_road]*length))
            
    road_train, road_test = normalize(np.array(road_train), np.array(road_test))


    # Demographics data
    demo_data = pd.read_csv(f'{data_path}/demographic/{city}-{year}-{level}.csv').drop(columns=['index'])
    demo_train, demo_test = [], []

    location_demo = []
    for location in selected_areas:
        location_demo.append(demo_data[location].to_numpy())

    for i in range(length, 721-n_steps):
        if i <= (21*24):
            demo_train.append(np.array([location_demo]*length))
        else:
            demo_test.append(np.array([location_demo]*length))
    demo_train, demo_test = normalize(np.array(demo_train), np.array(demo_test))


    # POI data
    poi_data = pd.read_csv(f'{data_path}/poi/{city}-{year}-{level}.csv').drop(columns=['location'])
    poi_train, poi_test = [], []

    location_poi = []
    for location in selected_areas:
        location_poi.append(poi_data[location].to_numpy())

    for i in range(length, 721-n_steps):
        if i <= (21*24):
            poi_train.append(np.array([location_poi]*length))
        else:
            poi_test.append(np.array([location_poi]*length))
            
    poi_train, poi_test = normalize(np.array(poi_train), np.array(poi_test))


    # Traffic volumes
    volume_data = pd.read_csv(f'{data_path}/traffic_volume/{city}-{level}-{year}.csv').drop(columns=['date', 'hour'])
    volume_train, volume_test = [], []

    for i in range(length, 721-n_steps):
        if i <= (21*24):
            volume_train.append(volume_data.iloc[i-length:i, :n_districts].to_numpy())
        else:
            volume_test.append(volume_data.iloc[i-length:i, :n_districts].to_numpy())

    volume_train, volume_test = normalize(np.array(volume_train), np.array(volume_test))


    # Calendar
    calendar_data = pd.read_csv(f'{data_path}/calendar/calendar-{city}-{year}-{level}.csv')
    calendar_train, calendar_test = [], []
    
    location_calendar = []
    for location in selected_areas:
        location_calendar.append(calendar_data[calendar_data['location'] == location].iloc[:, 1:].to_numpy())

    location_calendar = np.concatenate(location_calendar, axis=1)

    for i in range(length, 721-n_steps):
        if i <= (21*24):
            calendar_train.append(location_calendar[i-length:i])
        else:
            calendar_test.append(location_calendar[i-length:i])
            
    calendar_train, calendar_test = normalize(np.array(calendar_train).reshape(len(calendar_train), length, n_districts, -1), np.array(calendar_test).reshape(len(calendar_test), length, n_districts, -1))
    
    # Match Shape
    risk_train = risk_train[:,:,:,None]
    risk_test = risk_test[:,:,:,None]
    volume_train = volume_train[:,:,:,None]
    volume_test = volume_test[:,:,:,None]

    return {
        'risk': [risk_train, risk_test],
        'road': [road_train, road_test],
        'poi': [poi_train, poi_test],
        'demo': [demo_train, demo_test],
        'weather': [weather_train, weather_test],
        'calendar': [calendar_train, calendar_test],
        'volume': [volume_train, volume_test],
        'dtg': [dtg_train, dtg_test],
        'y': [y_train, y_test],
        'selected_areas': selected_areas,
    }
