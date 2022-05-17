import json
import streamlit as st
import pandas as pd
import numpy as np
import pgeocode
import requests

from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily
from sklearn.linear_model import LinearRegression
from configparser import ConfigParser

# set the page configuration
st.set_page_config(page_title="Crop Insurance Calculator", page_icon="👨‍🌾", layout="centered", initial_sidebar_state="expanded", menu_items=None)

# load configurations from config.ini file
config = ConfigParser()
config.read('config.ini')
start_year = int(config.get('configuration', 'start_year'))
end_year = int(config.get('configuration', 'end_year'))
start_month = int(config.get('configuration', 'start_month'))
end_month = int(config.get('configuration', 'end_month'))
prcp_days = int(config.get('configuration', 'prcp_days'))
prcp_amount = float(config.get('configuration', 'prcp_amount'))

# location options with postal codes
dict_locations = {
    'Hamburg':'20095',
    'Kusterdingen':'72127',
    'Munich':'80995', 
    'Stuttgart':'70173'
}

def changed_postal_code():
    try:
        nomi = pgeocode.Nominatim('de')
        # get lat and lon with postal code
        postalcode_data = nomi.query_postal_code(st.session_state['postal_code'])

        # write lat and lon in form
        if(str(postalcode_data['latitude'])!="nan" and str(postalcode_data['longitude'])!="nan"):
            st.session_state["lat"] = str(postalcode_data['latitude'])
            st.session_state["lon"] = str(postalcode_data['longitude'])
        else:
            st.session_state["lat"] = ""
            st.session_state["lon"] = ""
            st.error("Postal code not valid.")
    except Exception:
        st.error("Couldn't get the geocoordinates from the postal code. An internal error occured. Please try it later again.")

def changed_location():
    # change postal code based on select input
    if(location=='Hamburg'):
        st.session_state["postal_code"] = '20095'
    elif(location=='Kusterdingen'):
        st.session_state["postal_code"] = '72127'
    elif(location=='Munich'):
        st.session_state["postal_code"] = '80995'
    else: #stuttgart
        st.session_state["postal_code"] = '70173'

def get_elevation(lat, lon):
    # gets elevation of certain geocoordinates
    try:
        url = "https://api.opentopodata.org/v1/eudem25m?locations=" + str(lat) + "," + str(lon)
        response = requests.get(url).text
        response_info = json.loads(response)
        return response_info["results"][0]["elevation"]
    except Exception:
        st.error("Error while trying to get the elevation.")

st.sidebar.title('Crop Insurance Calculator')
st.sidebar.write('Welcome to the future of calculating crop insurance premiums. This example calculator is for open strawberry fields and focuses on the factor heavy rainfall.')

# user input fields
harvest = st.sidebar.slider(label='average harvest in last 5 years', min_value=5000, max_value=1000000, value=5000)
location = st.sidebar.selectbox(label='Location', options=['Hamburg', 'Kusterdingen', 'Munich', 'Stuttgart'])
postal_code = st.sidebar.text_input(label='Postal Code', value=dict_locations[location], key="postal_code", disabled=True)
changed_postal_code() # in order to autofill latitude and longitude
latitude = st.sidebar.text_input(label='Latitude', key='lat', disabled=True)
longitude = st.sidebar.text_input(label='Longitude', key='lon', disabled=True)    

def calculate():
    try:
        # get lat and lon from text input fields
        lat = float(latitude)
        lon = float(longitude)

        # get elevation
        elevation = get_elevation(lat, lon)

        # factor that later influences price based on elevation
        factor_upcharge_elevation = 1

        # Create Point for entered postal code
        location = Point(lat, lon)
    except Exception:
        st.error("Error occured. Make sure that lat and lon are in the right format.")    

    # create empty dataframe that is later used for the prediction
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

        # if there was a precipiation of over x (value in config file) it will definitely be a harvest loss
        data['loss'] = np.where(data['10dPrcp']>prcp_amount, 1, 0)
        
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

    # for some earlier years there is no avg. temp. -> avg. temperature-0.6 (global warming) is used
    mean_tavg = data['tavg'].mean() - 0.6
    df[['tavg']] = df[['tavg']].fillna(value=mean_tavg)

    # year is becoming an extra column to use it for linear regression
    df['year'] = df['time'].dt.year
    del df['time']
    df['n'] = df['year']
    df = df.set_index('n')

    # remove columns with low correlation to harvest loss
    del df['prcp']
    del df['tmax']

    # display results
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
    
    # making a prediction
    new_situation_predict = model.predict(np.array([[2022, 18], [2100, 18]]))
    st.write('Estimated risk for the next years until 2100:')
    st.line_chart(new_situation_predict)

    # if there's a low elevation it has an impact on the premium
    if elevation < 20:
        st.warning('There is a higher risk for floodings due to low elevation.')
        factor_upcharge_elevation = 1.1
    else:
        st.success('There is NO higher risk for floodings due to low elevation.')
    
    # display calculated premium and balloons
    new_premium_text = 'Calculated premium with AI: ' + str(round(harvest*new_situation_predict[0]*factor_upcharge_elevation, 2)) + ' €'
    st.subheader(new_premium_text)
    st.balloons()

st.sidebar.button('Calculate', on_click=calculate)