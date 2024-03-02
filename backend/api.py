import base64
from io import BytesIO

import fastapi
import numpy
import numpy as np
import pandas as pd
import geopandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import uvicorn
from cartopy import crs as ccrs
import openmeteo_requests
import requests_cache
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from retry_requests import retry
from geodatasets import get_path
import contextily as cx
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

import predict

app = fastapi.FastAPI()
origins = [
    "http://fire.adlemx.ru",
    "https://fire.adlemx.ru",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/get_fires_predition")
def get_fires(date_iso: str, accuracy:int=12):
    key = '0255e0228de18dd8232ab3bee2ba070a'
    cords = [15, 40, 17, 42]
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    print(",".join(map(str, cords)))
    area_url = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_NOAA20_NRT/{",".join(map(str, cords))}/1/{date_iso}'
    print(area_url)
    df_area = pd.read_csv(area_url)
    path = get_path("naturalearth.land")
    world = geopandas.read_file(path)
    gdf = geopandas.GeoDataFrame(
        df_area, geometry=geopandas.points_from_xy(df_area.longitude, df_area.latitude), crs="EPSG:4326"
    )

    lats = []
    longs = []

    step_longitude = (cords[2] - cords[0]) / (accuracy)
    step_latitude = (cords[3] - cords[1]) / (accuracy)
    for i in range(accuracy):
        for j in range(accuracy):
            longs.append(cords[0] + step_longitude * i)
            lats.append(cords[1] + step_latitude * j)
    cords_array = [(lats[i], longs[i]) for i in range(accuracy ** 2)]
    temps = []
    for i in range(0, len(cords_array), accuracy):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": [cords_array[i + j][0] for j in range(accuracy)],
            "longitude": [cords_array[i + j][1] for j in range(accuracy)],
            "start_date": {date_iso},
            "end_date": {date_iso},
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
                date_iso.split("-")[2], date_iso.split("-")[1],
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
    temps_n = np.array(temps)
    temps_n.shape = (-1, accuracy)
    plt.style.use('_mpl-gallery-nogrid')
    X, Y = np.meshgrid(np.linspace(cords[0], cords[2], accuracy),
                       np.linspace(cords[1], cords[3], accuracy))
    Z = temps_n
    ax = world.plot(figsize=(11, 10), alpha=0, aspect=1)
    contour = ax.contourf(X, Y, Z, alpha=0.6, cmap=mpl.colormaps["plasma"], levels=100)
    ax.set_xlim([cords[0], cords[2]])
    ax.set_ylim([cords[1], cords[3]])
    ax.axis('off')
    gdf.plot(ax=ax, color="red", markersize=10, aspect=1)
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(contour, ticks=[Z.min(), Z.max()], aspect=200, cax=cax1)
    cbar.ax.set_yticklabels(['MIN', 'MAX'])
    cx.add_basemap(ax, crs=gdf.crs)
    imgdata = BytesIO()
    ax.figure.savefig(imgdata, dpi=60, format='png')
    imgdata.seek(0)
    return StreamingResponse(imgdata, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9600, root_path=".")