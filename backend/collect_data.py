import numpy as np
import pandas as pd
import requests_cache
from retry_requests import retry
from utils import convert_time
import openmeteo_requests
path = "2022-07-08.csv"
df_area = pd.read_csv(path)
# df_area = pd.read_csv("2022-07-06-big2.csv")
df_area['acq_time'] = df_area['acq_time'].apply(convert_time)
df_area['acq_month'] = df_area['acq_date'].apply(lambda date: int(date.split("-")[1]))
df_area['acq_day'] = df_area['acq_date'].apply(lambda date: int(date.split("-")[2]))

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def get_temperature(date, time, lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": ["relative_humidity_2m", "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm",
                   "vapour_pressure_deficit", "temperature_2m", "precipitation", "wind_speed_10m", "dew_point_2m"]
    }
    response = openmeteo.weather_api(url, params=params)[0]
    hourly = response.Hourly()
    daily = response.Daily()
    hours = int(time.split(":")[0])-3
    relative_humidity = hourly.Variables(0).ValuesAsNumpy()[hours]
    soil_temperature = hourly.Variables(1).ValuesAsNumpy()[hours]
    soil_moisture = hourly.Variables(2).ValuesAsNumpy()[hours]
    temperature = hourly.Variables(4).ValuesAsNumpy()[hours]
    pressure_of_vapour = hourly.Variables(3).ValuesAsNumpy()[hours]
    precipitation = hourly.Variables(5).ValuesAsNumpy()[hours]
    wind = hourly.Variables(6).ValuesAsNumpy()[hours]
    dew_point = hourly.Variables(7).ValuesAsNumpy()[hours]
    return (temperature,
            relative_humidity,
            pressure_of_vapour,
            precipitation,
            soil_moisture,
            soil_temperature,
            wind,
            dew_point
            )


df_area[['temperature', "humidity", "vapour_pressure", "precipitation", "soil_moisture", "soil_temperature",
         "wind", "dew_point"]] \
    = (
    df_area.apply(lambda x: get_temperature(x.acq_date, x.acq_time, x.latitude, x.longitude),
                  axis=1,
                  result_type="expand"))
print(df_area.head())

df_area.to_csv( path + "-with-data.csv", index=False)