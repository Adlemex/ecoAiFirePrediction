import matplotlib
import numpy
import numpy as np
from matplotlib.contour import QuadContourSet
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import predict
from utils import convert_time, plot_color_gradients

key = '0255e0228de18dd8232ab3bee2ba070a'
#coords = [15, 41.2, 16, 42]
coords = [15, 40, 17, 42]
# best coords = [15, 40, 17, 42]
date = "2023-10-01"
num_weather_points = 10
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import openmeteo_requests
import requests_cache
from retry_requests import retry
from geodatasets import get_path
import contextily as cx

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)
print(",".join(map(str, coords)))
area_url = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_NOAA20_NRT/{",".join(map(str, coords))}/1/{date}'
print(area_url)
df_area = pd.read_csv(area_url)
df_area['acq_time'] = df_area['acq_time'].apply(convert_time)
path = get_path("naturalearth.land")
world = geopandas.read_file(path)
gdf = geopandas.GeoDataFrame(
    df_area, geometry=geopandas.points_from_xy(df_area.longitude, df_area.latitude), crs="EPSG:4326"
)

lats = []
longs = []

step_longitude = (coords[2] - coords[0]) / (num_weather_points)
step_latitude = (coords[3] - coords[1]) / (num_weather_points)
for i in range(num_weather_points):
    for j in range(num_weather_points):
        longs.append(coords[0] + step_longitude * i)
        lats.append(coords[1] + step_latitude * j)
cords_array = [(lats[i], longs[i]) for i in range(num_weather_points**2)]
temps = []
for i in range(0, len(cords_array), num_weather_points):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": [cords_array[i+j][0] for j in range(num_weather_points)],
        "longitude": [cords_array[i+j][1] for j in range(num_weather_points)],
        "start_date": {date},
        "end_date": {date},
        "hourly": ["relative_humidity_2m", "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm",
                   "vapour_pressure_deficit", "temperature_2m", "precipitation", "wind_speed_10m", "dew_point_2m"],
        "daily": ["temperature_2m_max", "temperature_2m_mean", "precipitation_sum"]
    }
    responses = openmeteo.weather_api(url, params=params)
    for response in responses:
        daily = response.Daily()
        hourly = response.Hourly()
        hours = 12 - 3
        relative_humidity = hourly.Variables(0).ValuesAsNumpy()[hours]
        soil_temperature = hourly.Variables(1).ValuesAsNumpy()[hours]
        soil_moisture = hourly.Variables(2).ValuesAsNumpy()[hours]
        temperature = hourly.Variables(4).ValuesAsNumpy()[hours]
        pressure_of_vapour = hourly.Variables(3).ValuesAsNumpy()[hours]
        precipitation = hourly.Variables(5).ValuesAsNumpy()[hours]
        wind = hourly.Variables(6).ValuesAsNumpy()[hours]
        dew_point = hourly.Variables(7).ValuesAsNumpy()[hours]
        daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy().mean()
        daily_temperature_2m_mean = daily.Variables(1).ValuesAsNumpy().mean()
        daily_precipitation_sum = daily.Variables(2).ValuesAsNumpy().mean()
        temps.append(predict.predict([
            response.Latitude(),
            response.Longitude(),
            date.split("-")[2], date.split("-")[1],
            temperature,
            relative_humidity,
            pressure_of_vapour,
            precipitation,
            soil_moisture,
            soil_temperature,
            wind,
            dew_point]))
        """temps.append((daily_temperature_2m_max*2)*
                     (1/hourly_relative_humidity_2m*10)*
                     (1/(hourly_soil_moisture_0_to_7cm+1))
                     *(hourly_soil_temperature_0_to_7cm**2)*
                     (1/(100*daily_precipitation_sum+1)))"""
        # temps.append(hourly_relative_humidity_2m)
temps_n = numpy.array(temps)
temps_n.shape = (-1, num_weather_points)
plt.style.use('_mpl-gallery-nogrid')
X, Y = np.meshgrid(np.linspace(coords[0], coords[2], num_weather_points),
                   np.linspace(coords[1], coords[3], num_weather_points))
Z = temps_n
levels = np.linspace(Z.min(), Z.max(), 7)
ax = world.plot(figsize=(11, 10), alpha=0)
contour = ax.contourf(X, Y, Z, alpha=0.6, cmap=matplotlib.colormaps["plasma"], levels=100)
ax.set_xlim([coords[0], coords[2]])
ax.set_ylim([coords[1], coords[3]])
ax.axis('off')
gdf.plot(ax=ax, color="red", markersize=10)
divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size="3%", pad=0.1)
cbar = plt.colorbar(contour, ticks=[Z.min(), Z.max()], aspect=200, cax=cax1)
cbar.ax.set_yticklabels(['MIN', 'MAX'])
cx.add_basemap(ax, crs=gdf.crs)
plt.show()
