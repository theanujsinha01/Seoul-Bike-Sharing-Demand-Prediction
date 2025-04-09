import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime

def initialize_model_and_scaler(model_path, sc_path):
    if os.path.exists(model_path) and os.path.exists(sc_path):
        model = pickle.load(open(model_path, "rb"))
        sc = pickle.load(open(sc_path, "rb"))
        return model, sc
    else:
        print("Model or Standard Scaler path not correct")
        return None, None

# Example usage:
model_path = r"models\xgb_model.pkl"
sc_path = r"models\scaler.pkl"

model, sc = initialize_model_and_scaler(model_path, sc_path)
if model and sc:
    print("Model and scaler loaded successfully")


def get_string_to_datetime(Date):
    dt = datetime.strptime(Date, "%d/%m/%Y")
    return {"day": dt.day, "month": dt.month, "year": dt.year, "weekday": dt.strftime("%A")}

def season_to_df(season):
    
    seasons_col = ['Spring', 'Summer', 'Winter']
    seasons_data = np.zeros((1,len(seasons_col)), dtype ="int")
    df_seasons = pd.DataFrame(seasons_data, columns = seasons_col)

    if season in seasons_col:
        df_seasons[season]=1
    return df_seasons 

def days_name(weekday):
    days_names = ['Monday', 'Saturday','Sunday', 'Thursday', 'Tuesday', 'Wednesday']
    days_name_data = np.zeros((1,len(days_names)), dtype="int")
    df_days = pd.DataFrame(days_name_data, columns=days_names)

    if weekday in days_names:
        df_days[weekday] = 1
    return df_days

def users_input():
    print("Enter correct information to predict Rented Bike count")


    Date = input("Date (dd/mm/yyyy): ")
    Hour = int(input("Hours (0-23):"))
    Temperature = float(input("Temprature in C :"))
    Humidity = float(input("Humidity:"))
    Wind_speed = float(input("Wind Speed :"))
    visibility = float(input("visibility :"))
    Solar_Radiation = float(input("Solar  radiation :"))
    Rainfall = float(input("Rainfall :"))
    Snowfall = float(input("Snowfall :"))
    seasons  = input("Seasons (Antum, Spring, Summer, Winter): ")
    Holiday = input("Holiday (Holiday/No Holiday): ")
    Functioning_Day = input("Functioning Day (Yes/No): ")
    holiday_dict = {"No Holiday":0, "Holiday":1}
    functioning_day_dict = {"No":0, "Yes":1}
    str_to_date = get_string_to_datetime(Date)

    u_input_list = [Hour,Temperature,Humidity,Wind_speed,visibility,Solar_Radiation,Rainfall,Snowfall ,
                holiday_dict[Holiday],functioning_day_dict[Functioning_Day], str_to_date["day"],str_to_date["month"],str_to_date["year"],]
    
    features_name = ['Hour', 'Temperature(°C)', 'Humidity(%)',
       'Wind speed (m/s)', 'Visibility (10m)', 'Solar Radiation (MJ/m2)',
       'Rainfall(mm)', 'Snowfall (cm)', 'Holiday', 'Functioning Day', 'Day',
        'Month', 'Year'
       ]
    df_u_input = pd.DataFrame([u_input_list], columns=features_name)
    seasons_df = season_to_df(seasons)
    days_name_df = days_name(str_to_date["weekday"])
    df_final_input = pd.concat([df_u_input,seasons_df,days_name_df ], axis=1)
    
    return df_final_input

def prediction():
    df = users_input()
    scaled_data = sc.transform(df)
    prediction =  model.predict(scaled_data)

    return prediction



prediction_final = prediction()
print(f"Rented Bike Demand is: {round(prediction_final.tolist()[0])}")

