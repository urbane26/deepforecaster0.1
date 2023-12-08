#import core pkgs
import streamlit as st
import streamlit_authenticator as stauth
import streamlit.components.v1 as stc
import base64

st.set_page_config('Forecaster',initial_sidebar_state='expanded')
st.set_option('deprecation.showPyplotGlobalUse', False)

#import supporting pkgs
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import numpy as np
from PIL import Image
import pathlib
from pathlib import Path
import glob
import time
import shutil
import tempfile
import os

#core plotting pkgs
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import plotly.express as px
import seaborn as sns
sns.set_theme(context='notebook', style='darkgrid', palette='deep')
from sklearn.metrics import mean_absolute_percentage_error

#current working path
path = os.getcwd()
abspath = path+'\\data'

#support functions
#function to save file in cwd
def saveuploadedfile(uploadedfile):
    with open(uploadedfile.name,'wb') as f:
        f.write(uploadedfile.getbuffer())
    return st.success('File Saved')

#function to zip files
@st.cache_data
def zipped(path):
    arc = shutil.make_archive('directory', format='zip', root_dir=path)
    return arc

#importing pkgs from other files
from preprocessing import datasetset_preprocessing_pipeline
from eda import forecastability , corranalysis ,shapanalysis, stationaritychk, DemandClassification
from forecaster import PatchTSTforecast,Prophetforecast,Nbeatsforecast,Nhitsforecaster,SESoptforecaster,\
    TemporalFusionTransformerforecaster,DilatedRNNforecaster,GRUforecaster,LSTMforecaster,DeepARforecaster

#initiating authintication
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate( config['credentials'],config['cookie']['name'],config['cookie']\
                                    ['key'],config['cookie']['expiry_days'],config['preauthorized'])
authenticator.login('Login', 'sidebar')
if st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'main', key='unique_key')
    st.success(f'Welcome *{st.session_state["name"]}*')

    st.title = 'Deep Forecaster'
    
    Menu = ["Home","Exploratry_Analysis","Forecaster","Helper"]
    Choice = st.sidebar.selectbox("Menu",Menu)
    if Choice == "Home":
        st.header('Hello!')
        st.header('Welcome to AI forecaster')
        st.subheader('Quick Start :')
        st.subheader('1) Upload univariate files in the following format : Replace "TIME" with ds and "Tagret var" with "y" and group variable by "unique_id"')
        st.subheader('2) Upload multivariate files in the following format : Replace "TIME" with ds,group variable by "unique_id" and "Tagret var" with "y" followed by features')
        st.subheader('3) Move to individual items by clicking the menu on the sidebar')
        st.subheader('4) Follow instructions on individual dropdowns by clicking the widgets')
        st.subheader('5) Use the button below to refresh if you are doing a new or re-run')
        if st.checkbox ('Make/Refresh project folders'):
            st.markdown("The current working directory is %s" % path)
            if os.path.exists(abspath):
                try:
                    shutil.rmtree(abspath)
                    os.mkdir(abspath)
                    os.mkdir(abspath+'\\plots')
                    os.mkdir(abspath+'\\stats')
                    os.mkdir(abspath+'\\input')
                    os.mkdir(abspath+'\\eda')
                except OSError:
                    st.write("Refreshing the directory %s failed" % abspath)
                else:
                    st.success("Successfully refreshed/created the directories")
            else:
                st.write('required folders did not exist...created these')
                os.mkdir(abspath)
                os.mkdir(abspath+'\\plots')
                os.mkdir(abspath+'\\stats')
                os.mkdir(abspath+'\\input')
                os.mkdir(abspath+'\\eda')

    elif Choice == "Exploratry_Analysis":

        data_file = st.file_uploader('Upload any csv file with the required format',type =['csv'])
        selection = st.radio('choose nature of file',["Univariate","Multivariate"])
        if selection == 'Univariate':
            if data_file is not None:
                df = pd.read_csv(data_file,parse_dates=True)
                st.dataframe(df)
                if np.array(df.shape)[0] < 10* np.array(df.shape)[1]:
                    st.write('Please note that your dataset is not prefectly suitable for ML in terms of size. To get\
                     good results,try to fetch more observations.You may proceed anyway')
                if np.array(df.shape)[0] < 1000:
                    st.write('Your data has less number of observations than what is ideal for deep learning based forecasting. Get more observations or resample.\
                     You may proceed anyway')
                saveuploadedfile(data_file)

                for i,g in df.groupby(df['unique_id']):
                    g.to_csv('data\input\{}.csv'.format(i), header=True, index_label='Index')
                    filenames = sorted(glob.glob('data\input\*.csv'))

                on = st.toggle('Activate EDA Run',key='activate')
                if on:
                    for f in filenames:
                        container = st.container()
                        print(f)
                        dataset = pd.read_csv(f)
                        shapes = dataset.shape
                        dir = f
                        a = Path(dir).stem
                        #demand classification
                        expr1 = DemandClassification(dataset)
                        container.write('for %s : {}'.format(expr1)%a)
                        with open('data\\stats\\classification_results%s.txt'%a,'w') as g:
                            g.write('for %s : {}'.format(expr1)%a+'\n')
                            g.close()
                        #Stationarity check
                        expr2 = stationaritychk(dataset['y'])
                        container.write('for %s : {}'.format(expr2)%a)
                        with open('data\\stats\\stationarity_results%s.txt'%a,'w') as h:
                            h.write('for %s : {}'.format(expr2)%a+'\n')
                            h.close()
                        #forecastability check
                        apn = forecastability(dataset['y'])
                        container.write('for %s : {}'.format(apn)%a)
                        with open('data\\stats\\apresults%s.txt'%a,'w') as k:
                            k.write(str(1-apn) + '  forecastability : lower is better'+'\n')
                            k.close()

                        prg = st.progress(0) 
                        for i in range(100): 
                            time.sleep(0.01) 
                            prg.progress(i+1)

                    directory = pathlib.Path(abspath)
                    zipped(abspath)
                    with open("directory.zip", "rb") as fp:
                        btn = st.download_button(
                        label="Download ZIP",
                        data=fp,
                        file_name="directory.zip",
                        mime="application/zip"
                        )

                    st.write('all output files have been zipped for download')                                    
        elif selection == 'Multivariate':
            if data_file is not None:
                df = pd.read_csv(data_file,parse_dates=True)
                st.dataframe(df)
                saveuploadedfile(data_file)
                collist = df.columns.to_list()

                for i,g in df.groupby(df['unique_id']):
                    g.to_csv('data\input\{}.csv'.format(i), header=True, index_label='Index')
                    filenames = sorted(glob.glob('data\input\*.csv'))

                on = st.toggle('Activate EDA Run',key='activate')
                if on:
                    for f in filenames:
                        print(f)
                        dataset = pd.read_csv(f)
                        shapes = dataset.shape
                        dir = f
                        a = Path(dir).stem
                        dataset = datasetset_preprocessing_pipeline(dataset)
                        #demand classification
                        expr1 = DemandClassification(dataset)
                        st.write('class for %s : {}'.format(expr1)%a)
                        with open('data\\stats\\classification_results%s.txt'%a,'w') as g:
                            g.write('class for %s : {}'.format(expr1)%a+'\n')
                            g.close()
                        #Stationarity check
                        expr2 = stationaritychk(dataset['y'])
                        st.write('for%s : {}'.format(expr2)%a)
                        with open('data\\stats\\stationarity_results%s.txt'%a,'w') as h:
                            h.write('for %s : {}'.format(expr2)%a+'\n')
                            h.close()
                        #forecastability check
                        apn = forecastability(dataset['y'])
                        st.write('Noise for%s : {}'.format(apn)%a)
                        with open('data\\stats\\apresults%s.txt'%a,'w') as k:
                            k.write(str(1-apn) + '  forecastability : lower is better'+'\n')
                            k.close()
                        #Correlation check
                        cor_matrix = corranalysis(dataset)
                        fig, ax = plt.subplots()
                        sns.heatmap(dataset.corr(),cmap="YlGnBu", annot=False, ax=ax)
                        st.write(fig)
                        cor_matrix.to_csv('data\\stats\\correlaion_marix%s.csv'%a)
                        #feature importance check
                        fig, ax = plt.subplots()
                        shapanalysis(dataset)
                        st.write(fig)
                        plt.savefig('data\\plots\\SHAP_summary_feature_importance%s.jpg'%a,bbox_inches ="tight")

                        prg = st.progress(0) 
                        for i in range(100): 
                            time.sleep(0.01) 
                            prg.progress(i+1)

                    directory = pathlib.Path(abspath)
                    zipped(abspath)
                    with open("directory.zip", "rb") as fp:
                        btn = st.download_button(
                        label="Download ZIP",
                        data=fp,
                        file_name="directory.zip",
                        mime="application/zip"
                        )

                
                featureset = st.multiselect('Select features for forecasting',collist)

        try:
            st.session_state.filenames = filenames
            st.session_state.featureset = featureset
        except:
            st.write('Waiting for forecaster to load...')    

    elif Choice == "Forecaster":
        try:
            featureset = st.session_state.featureset
            filenames = st.session_state.filenames
        except:
            st.write('waitng for features to lead...')
        st.subheader('FEATURES')
        try:
            st.write('These are the features you have selected for forecasting : {}'.format(featureset))
        except:
            st.write('Features have been loaded')

        st.header('Select periods ahead to forecast')
        hor = st.number_input('Enter forecast horizon',min_value=1,step =1,max_value = 24)
        
        data = {
            'Mode': ['Select one of the following...','Univariate Mode','Multivariate Mode']
        }

        dg = pd.DataFrame(data)
        selected_category = st.selectbox("Select Mode", dg['Mode'].unique())        
        filtered_df = dg[dg['Mode'] == selected_category] 

        if filtered_df.iloc[0,0] == 'Univariate Mode':
            data = {
            'Mode': ['Select one of the following...','PatchTST', 'NBEATS', 'NHITS','Prophet']
        }

            dm = pd.DataFrame(data)
            selected_category = st.selectbox("Select Algorithm", dm['Mode'].unique())        
            f_df = dm[dm['Mode'] == selected_category] 
        elif filtered_df.iloc[0,0] == 'Multivariate Mode':
            data = {
            'Mode': ['Select one of the following...','T-FUSION TRANSFORMER','LSTM','GRU','DilatedRNN','StemGNN','DeepAR']
        }

            dm = pd.DataFrame(data)
            selected_category = st.selectbox("Select Algorithm", dm['Mode'].unique())        
            f_df = dm[dm['Mode'] == selected_category]
        else:
            st.write('Please select a mode of forecasting')

        if st.button('Generate Forecast'):
            if f_df.iloc[0,0] == 'PatchTST':
                filenames = sorted(glob.glob('data\input\*.csv'))
                for f in filenames:
                    with st.spinner('Please wait while the forecast is calculated by unique id...'): 
                        dataset = pd.read_csv(f)  
                        dir = f
                        a = Path(dir).stem
                        patchtst_df,plot_df = PatchTSTforecast(dataset,hor)
                        patchtst_df.to_csv('data\\stats\\patchtst_forecast_output%s.csv'%a)
                        MAPE =  mean_absolute_percentage_error(plot_df['y'],plot_df['PatchTST'])       
                        st.write('%s The calculated forecast accuracy is {} percent'.format(np.round((100.00-MAPE*100,2)))%a)
                    prg = st.progress(0) 
                    for i in range(100): 
                        time.sleep(0.01) 
                        prg.progress(i+1) 
                st.success("process completed successfully")
                zipped(abspath)                
                with open("directory.zip", "rb") as fp:
                        btn = st.download_button(
                        label="Download ZIP",
                        data=fp,
                        file_name="directory.zip",
                        mime="application/zip"
                        )

            elif f_df.iloc[0,0] == 'NBEATS':
                filenames = sorted(glob.glob('data\input\*.csv'))
                for f in filenames:
                    with st.spinner('Please wait while the forecast is calculated by unique id...'): 
                        dataset = pd.read_csv(f)  
                        dir = f
                        a = Path(dir).stem
                        nbeats_df,plot_df = Nbeatsforecast(dataset,hor)
                        nbeats_df.to_csv('data\\stats\\NBEATS_forecast_output%s.csv'%a)
                        MAPE =  mean_absolute_percentage_error(plot_df['y'],plot_df['NBEATS'])       
                        st.write('%s The calculated forecast accuracy is {} percent'.format(np.round((100.00-MAPE*100,2)))%a)
                    prg = st.progress(0) 
                    for i in range(100): 
                        time.sleep(0.01) 
                        prg.progress(i+1) 
                st.success("process completed successfully")
                zipped(abspath)                
                with open("directory.zip", "rb") as fp:
                        btn = st.download_button(
                        label="Download ZIP",
                        data=fp,
                        file_name="directory.zip",
                        mime="application/zip"
                        )

            elif f_df.iloc[0,0] == 'NHITS':
                filenames = sorted(glob.glob('data\input\*.csv'))
                for f in filenames:
                    with st.spinner('Please wait while the forecast is calculated by unique id...'): 
                        dataset = pd.read_csv(f)  
                        dir = f
                        a = Path(dir).stem
                        nhits_df,plot_df = Nhitsforecaster(dataset,hor)
                        nhits_df.to_csv('data\\stats\\NHITS_forecast_output%s.csv'%a)
                        MAPE =  mean_absolute_percentage_error(plot_df['y'],plot_df['NHITS'])       
                        st.write('%s The calculated forecast accuracy is {} percent'.format(np.round((100.00-MAPE*100,2)))%a)
                    prg = st.progress(0) 
                    for i in range(100): 
                        time.sleep(0.01) 
                        prg.progress(i+1) 
                st.success("process completed successfully")
                zipped(abspath)                
                with open("directory.zip", "rb") as fp:
                        btn = st.download_button(
                        label="Download ZIP",
                        data=fp,
                        file_name="directory.zip",
                        mime="application/zip"
                        )
            elif f_df.iloc[0,0] == 'Prophet':
                    filenames = sorted(glob.glob('data\input\*.csv'))
                    for f in filenames:
                        with st.spinner('Please wait while the forecast is calculated by unique id...'):
                            dataset = pd.read_csv(f)  
                            dir = f
                            a = Path(dir).stem
                            forecast,fig1,fig2,MAPE = Prophetforecast(dataset,hor) 
                            st.write('%s The calculated forecast accuracy is {} percent'.format(np.round((100.00-MAPE*100,2)))%a)               
                            forecast.to_csv('data\\stats\\fbprophet_forecast_output%s.csv'%a)
                            fig, ax = plt.subplots()
                            st.write(fig1)
                            plt.savefig('data\\plots\\FbProphet_forecast_plot%s.jpg'%a,bbox_inches ="tight")
                            fig, ax = plt.subplots()
                            st.write(fig2)
                            st.write('FbProphet_forecast_plot%s'%a)
                            plt.savefig('data\\plots\\FbProphet_comp_plot%s.jpg'%a,bbox_inches ="tight")
                        prg = st.progress(0) 
                        for i in range(100): 
                            time.sleep(0.01) 
                            prg.progress(i+1) 
                    st.success("process completed successfully")
                    zipped(abspath) 
                    with open("directory.zip", "rb") as fp:
                        btn = st.download_button(
                        label="Download ZIP",
                        data=fp,
                        file_name="directory.zip",
                        mime="application/zip")
            elif f_df.iloc[0,0] == 'T-FUSION TRANSFORMER':
                    filenames = sorted(glob.glob('data\input\*.csv'))
                    for f in filenames:
                        with st.spinner('Please wait while the forecast is calculated by unique id...'):
                            dataset = pd.read_csv(f)  
                            dir = f
                            a = Path(dir).stem
                            forecast = TemporalFusionTransformerforecaster(dataset,hor)              
                            forecast.to_csv('data\\stats\\TFT_forecast_output%s.csv'%a)
                        prg = st.progress(0) 
                        for i in range(100): 
                            time.sleep(0.01) 
                            prg.progress(i+1) 
                    st.success("process completed successfully")
            elif f_df.iloc[0,0] == 'DeepAR':
                    filenames = sorted(glob.glob('data\input\*.csv'))
                    for f in filenames:
                        with st.spinner('Please wait while the forecast is calculated by unique id...'):
                            dataset = pd.read_csv(f)  
                            dir = f
                            a = Path(dir).stem
                            forecast = DeepARforecaster(dataset,hor)              
                            forecast.to_csv('data\\stats\\DeepAR_forecast_output%s.csv'%a)
                        prg = st.progress(0) 
                        for i in range(100): 
                            time.sleep(0.01) 
                            prg.progress(i+1) 
                    st.success("process completed successfully")
                    zipped(abspath) 
                    with open("directory.zip", "rb") as fp:
                        btn = st.download_button(
                        label="Download ZIP",
                        data=fp,
                        file_name="directory.zip",
                        mime="application/zip")
            elif f_df.iloc[0,0] == 'LSTM':
                    filenames = sorted(glob.glob('data\input\*.csv'))
                    for f in filenames:
                        with st.spinner('Please wait while the forecast is calculated by unique id...'):
                            dataset = pd.read_csv(f)  
                            dir = f
                            a = Path(dir).stem
                            forecast = LSTMforecaster(dataset,hor)              
                            forecast.to_csv('data\\stats\\LSTM_forecast_output%s.csv'%a)
                        prg = st.progress(0) 
                        for i in range(100): 
                            time.sleep(0.01) 
                            prg.progress(i+1)
                    st.success("process completed successfully")
                    zipped(abspath) 
                    with open("directory.zip", "rb") as fp:
                        btn = st.download_button(
                        label="Download ZIP",
                        data=fp,
                        file_name="directory.zip",
                        mime="application/zip")
            elif f_df.iloc[0,0] == 'DilatedRNN':
                    filenames = sorted(glob.glob('data\input\*.csv'))
                    for f in filenames:
                        with st.spinner('Please wait while the forecast is calculated by unique id...'):
                            dataset = pd.read_csv(f)  
                            dir = f
                            a = Path(dir).stem
                            forecast = DilatedRNNforecaster(dataset,hor)              
                            forecast.to_csv('data\\stats\\DRNN_forecast_output%s.csv'%a)
                        prg = st.progress(0) 
                        for i in range(100): 
                            time.sleep(0.01) 
                            prg.progress(i+1)
                    st.success("process completed successfully")
                    zipped(abspath) 
                    with open("directory.zip", "rb") as fp:
                        btn = st.download_button(
                        label="Download ZIP",
                        data=fp,
                        file_name="directory.zip",
                        mime="application/zip")
            elif f_df.iloc[0,0] == 'GRU':
                    filenames = sorted(glob.glob('data\input\*.csv'))
                    for f in filenames:
                        with st.spinner('Please wait while the forecast is calculated by unique id...'):
                            dataset = pd.read_csv(f)  
                            dir = f
                            a = Path(dir).stem
                            forecast = GRUforecaster(dataset,hor)              
                            forecast.to_csv('data\\stats\\GRU_forecast_output%s.csv'%a)
                        prg = st.progress(0) 
                        for i in range(100): 
                            time.sleep(0.01) 
                            prg.progress(i+1)
                    st.success("process completed successfully")
                    zipped(abspath) 
                    with open("directory.zip", "rb") as fp:
                        btn = st.download_button(
                        label="Download ZIP",
                        data=fp,
                        file_name="directory.zip",
                        mime="application/zip")
            else:
                st.markdown('NOT IMPLEMENTED YET')

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')