import streamlit as st
import pandas as pd
import numpy as np
import pgeocode

from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily
from meteostat import Stations
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from configparser import ConfigParser

def changedPostalCode():
    nomi = pgeocode.Nominatim('de')
    # get lat and lon with postal code
    postalcode_data = nomi.query_postal_code(st.session_state['postalCode'])

    # write lat and lon in form
    st.session_state["lat"] = str(postalcode_data['latitude'])
    st.session_state["lon"] = str(postalcode_data['longitude'])

def calculate():
    nomi = pgeocode.Nominatim('de')

    #nomi.query_postal_code(float(postalCode))
    postalcode_data = nomi.query_postal_code(postalCode)

    # Create Point for entered postal code
    location = Point(postalcode_data['latitude'], postalcode_data['longitude'])

    df = pd.DataFrame()

    # Create datset with data from meteostat
    for year in range(start_year, end_year):
        
        # Set time period
        start = datetime(year, start_month, 1)
        end = datetime(year, end_month, 31)

        # Get daily data
        data = Daily(location, start, end)
        data = data.fetch()
        # add for each row an entry with the sum of precipitation of last 8 days (config file)
        data['10dPrcp'] = data['prcp'].rolling(prcp_days).sum()
        # if there was a precipiation of over 95mm (config file) it will definitely be a harvest loss
        data['loss'] = np.where(data['10dPrcp']>prcp_amount, 1, 0)
        # for some years or locations there is no avg. temp. -> temperature of 15 (config file) is used
        data[['tavg']] = data[['tavg']].fillna(value=avg_temperature)
        
        # calculate tavg for entire year
        mean_tavg = data['tavg'].mean()
        del data['tavg']
        data['tavg'] = mean_tavg
        
        # calculate tmin for entire year
        mean_tmin = data['tmin']
        del data['tmin']
        data['tmin'] = mean_tmin
        
        # calculate tmax for entire year
        mean_tmax = data['tmax']
        del data['tmax']
        data['tmax'] = mean_tmax
        
        # wind direction is removed because there is often no data about it
        del data['wdir']
        
        # wind speed is removed because there is often no data about it
        del data['wspd']
        
        # wind peek gust is removed because there is often no data about it
        del data['wpgt']
        
        # air pressure is removed because there is often no data about it
        del data['pres']

        # # sun time is removed because there is often no data about it
        del data['tsun']
        
        # snow data is entirely removed because because there is often no data about it
        del data['snow']
        
        data.reset_index(inplace=True)
        
        # analysing missing values
        data.isnull().sum()

        # for each year only one entry is going to be used (the one with the highest cumulated precipitation in the last days)
        df = df.append(data.loc[data['10dPrcp'].argmax()])

    # year is becoming an extra column to use it for linear regression
    df['year'] = df['time'].dt.year
    del df['time']
    df['n'] = df['year']
    df = df.set_index('n')

    # remove columns with low correlation to harvest loss
    del df['prcp']
    del df['tmax']

    st.write('Harvest losses since 1900')
    st.bar_chart(df['loss'])
    st.write('Heavy rainfall since 1900')
    st.bar_chart(df['10dPrcp'])
    st.write('Average seasonal temperature since 1900')
    st.line_chart(df['tavg'])

    # displaying a plot
    df.plot(y=['loss'])
    plt.xlabel("Year")
    plt.ylabel("Harvest loss")
    plt.show()

    # set X and y variables
    X = df[['year', 'tavg']]
    y = df['loss']

    # doing linear regression
    model = LinearRegression()
    model.fit(X, y)
    #model.intercept_
    #model.coef_

    model_results = pd.DataFrame(model.coef_, X.columns, columns=['Coeffcicients'])

    # making a prediction
    new_situation_predict = model.predict(np.array([[2022, 18], [2122, 18]]))
    st.write('Estimated risk for the next 100 years (from 2022 - 2122):')
    st.line_chart(new_situation_predict)
    st.write('Calculated premium with AI: ', harvest*new_situation_predict[0])


# load configurations from config.ini file
config = ConfigParser()
config.read('config.ini')
start_year = int(config.get('configuration', 'start_year'))
end_year = int(config.get('configuration', 'end_year'))
start_month = int(config.get('configuration', 'start_month'))
end_month = int(config.get('configuration', 'end_month'))
avg_temperature = float(config.get('configuration', 'avg_temperature'))
prcp_days = int(config.get('configuration', 'prcp_days'))
prcp_amount = float(config.get('configuration', 'prcp_amount'))



st.title('Crop Insurance Calculator')
st.write('Welcome to the future of calculating crop insurance premiums. This example calculator is for open strawberry fields and focuses on the factor heavy rainfall.')
area = st.slider('area in m^2', min_value=5000, max_value=100000, value=5000)
harvest = st.slider('average harvest in last 5 years', min_value=5000, max_value=1000000, value=5000)
postalCode = st.text_input('Postal Code', on_change=changedPostalCode, key="postalCode")
latitude = st.text_input('Latitude', key='lat')
longitude = st.text_input('Longitude', key='lon')
st.write('Calculated premium without AI: ', harvest*0.07)
st.button('Calculate', on_click=calculate)



