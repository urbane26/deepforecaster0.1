#import core pkgs
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from neuralforecast.core import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, PatchTST,TFT,DeepAR,LSTM,GRU,DilatedRNN,StemGNN
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import DeepAR
from neuralforecast.models import StemGNN
from neuralforecast.losses.pytorch import MAE
from statsforecast import StatsForecast
from statsforecast.models import SimpleExponentialSmoothingOptimized as SESOpt
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from neuralforecast.auto import AutoNHITS

#import supporting pkgs
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error as mape


#supporting functions



#main functions for forecasting

#UNIVARIATE

def PatchTSTforecast(dataset: pd.DataFrame,hor:int):
    dataset['ds'] = pd.to_datetime(dataset['ds'])
    frequency = pd.infer_freq(dataset['ds'])
    train_set, test_set= np.split(dataset, [int(.80 *len(dataset))])
    horizon = len(test_set)+hor
    m = PatchTST(h=horizon,
                 input_size=104,
                 patch_len=24,
                 stride=24,
                 revin=True,
                 hidden_size=16,
                 n_heads=4,
                 scaler_type='robust',
                 loss=DistributionLoss(distribution='StudentT', level=[80, 90]),
                 learning_rate=1e-3,
                 max_steps=500,
                 val_check_steps=2,
                 early_stop_patience_steps=0,start_padding_enabled=True)
    nf = NeuralForecast(models=[m], freq=frequency,local_scaler_type='standard')
    nf.fit(df = train_set,val_size=0)
    patchtst_df = nf.predict().reset_index()
    plot_df = pd.merge(test_set, patchtst_df, on=["ds"])
    return patchtst_df,plot_df

    
def Prophetforecast(dataset: pd.DataFrame,hor:int):
    dataset['ds'] = pd.to_datetime(dataset['ds'])
    frequency = pd.infer_freq(dataset['ds'])
    data_set = dataset[['ds','y']]
    k = int(.80 *len(dataset))
    train = data_set.iloc[:k]
    test = data_set.iloc[k:]
    m = Prophet()
    m.fit(train)
    future_t = m.make_future_dataframe(periods=len(test),freq = frequency)
    prophetforecast_t = m.predict(future_t)
    y_hat = prophetforecast_t.iloc[-len(test):]
    MAPE = mape(test['y'], y_hat['yhat'])
    z = Prophet()
    z.fit(data_set)
    future = z.make_future_dataframe(periods=hor,freq = frequency)
    prophetforecast = z.predict(future)
    fig1 = z.plot(prophetforecast)
    fig2 = z.plot_components(prophetforecast)
    return prophetforecast,fig1,fig2,MAPE


def Nbeatsforecast(dataset: pd.DataFrame,hor:int):
    dataset['ds'] = pd.to_datetime(dataset['ds'])
    frequency = pd.infer_freq(dataset['ds'])
    train_set, test_set= np.split(dataset, [int(.80 *len(dataset))])
    horizon = len(test_set)+hor
    m= NBEATS(h=horizon,input_size=2*horizon,max_steps=50,start_padding_enabled=True)
    nf = NeuralForecast(models=[m], freq=frequency,local_scaler_type='standard')
    nf.fit(df = train_set)
    nbeats_df = nf.predict().reset_index()
    plot_df = pd.merge(test_set, nbeats_df, on=["ds"])
    return nbeats_df,plot_df


def Nhitsforecaster(dataset: pd.DataFrame,hor:int):
    dataset['ds'] = pd.to_datetime(dataset['ds'])
    frequency = pd.infer_freq(dataset['ds'])
    train_set, test_set= np.split(dataset, [int(.80 *len(dataset))])
    horizon = len(test_set)+hor
    m= NHITS(h=horizon,input_size=2*horizon,max_steps=50,start_padding_enabled=True)
    nf = NeuralForecast(models=[m], freq=frequency,local_scaler_type='standard')
    nf.fit(df = train_set)
    nhits_df = nf.predict().reset_index()
    plot_df = pd.merge(test_set, nhits_df, on=["ds"])
    return nhits_df,plot_df

def SESoptforecaster(dataset: pd.DataFrame,hor:int):
    dataset['ds'] = pd.to_datetime(dataset['ds'])
    frequency = pd.infer_freq(dataset['ds'])
    models = [SESOpt()]
    sf = StatsForecast(df=dataset,models=models,freq=frequency,n_jobs=-1)
    sf.fit(dataset)
    forecasts = sf.predict(hor, level=[95])
    return forecasts


#MULTIVARIATE

def TemporalFusionTransformerforecaster(dataset: pd.DataFrame,hor:int,a:str):
    featuredataset = dataset.drop(['ds','y','unique_id'],axis = 1)
    featurelist = featuredataset.columns.to_list()
    exog_fut_data = []
    for feature in featurelist:
        dx = dataset[['ds',feature]]
        frequency = pd.infer_freq(dataset['ds'])
        dx.set_index('ds',inplace=True)        
        fitted_model = ExponentialSmoothing(dx[feature],trend = 'additive',seasonal= 'additive',seasonal_periods=12,freq= frequency).fit()
        dz = fitted_model.forecast(12)
        dz = dz.to_frame()
        dz = dz.reset_index()
        dz = dz.rename(columns= {'index': 'ds',0: feature})
        exog_fut_data.append(dz)
    exog = [df.set_index('ds') for df in exog_fut_data]
    exog_future = pd.concat(exog, axis=1)
    exog_future = exog_future.reset_index()
    exog_future.insert(0, "unique_id", a) 
    futr_dfx = exog_future.reset_index()
    static_dfx = dataset['unique_id'].to_frame()
    #dataset = dataset.rename(columns={'Index':'count'})
    dataset['ds'] = pd.to_datetime(dataset['ds'])
    hist_df = dataset#.drop('count',axis = 1)
    hist_df['ds'] = pd.to_datetime(hist_df['ds'])
    futr_dfx = futr_dfx.rename(columns={'index':'count'})
    futr_dfx2 = futr_dfx.drop(['count'],axis = 1)
    futr_exog_l =   exog_future.drop(['unique_id','ds'],axis=1).columns.to_list()                   
    model = TFT(h=hor,input_size=48,hidden_size=20,loss=DistributionLoss(distribution='Normal', level=[80, 90]),learning_rate=0.005,\
                futr_exog_list=futr_exog_l,max_steps=500,val_check_steps=10,early_stop_patience_steps=0,\
                    scaler_type='robust',windows_batch_size=None,enable_progress_bar=True)
    fcst = NeuralForecast(models=[model], freq=frequency)
    fcst.fit(df=hist_df,val_size=0)
    tftforecasts = fcst.predict(futr_df=futr_dfx2)
    return tftforecasts


def DilatedRNNforecaster(dataset: pd.DataFrame,hor:int,a:str):
    featuredataset = dataset.drop(['ds','y','unique_id'],axis = 1)
    featurelist = featuredataset.columns.to_list()
    exog_fut_data = []
    for feature in featurelist:
        dx = dataset[['ds',feature]]
        frequency = pd.infer_freq(dataset['ds'])
        dx.set_index('ds',inplace=True)        
        fitted_model = ExponentialSmoothing(dx[feature],trend = 'additive',seasonal= 'additive',seasonal_periods=12,freq= frequency).fit()
        dz = fitted_model.forecast(hor)
        dz = dz.to_frame()
        dz = dz.reset_index()
        dz = dz.rename(columns= {'index': 'ds',0: feature})
        exog_fut_data.append(dz)
    exog = [df.set_index('ds') for df in exog_fut_data]
    exog_future = pd.concat(exog, axis=1)
    exog_future = exog_future.reset_index()
    exog_future.insert(0, "unique_id", a) 
    futr_dfx = exog_future.reset_index()
    static_dfx = dataset['unique_id'].to_frame()
    #dataset = dataset.rename(columns={'Index':'count'})
    dataset['ds'] = pd.to_datetime(dataset['ds'])
    hist_df = dataset#.drop('count',axis = 1)
    hist_df['ds'] = pd.to_datetime(hist_df['ds'])
    futr_dfx = futr_dfx.rename(columns={'index':'count'})
    futr_dfx = futr_dfx.drop('count',axis = 1)
    futr_exog_l =   exog_future.drop(['unique_id','ds'],axis=1).columns.to_list()                   
    model = DilatedRNN(h=hor,input_size=-1,loss=DistributionLoss(distribution='Normal', level=[80, 90]),scaler_type='robust',\
                       encoder_hidden_size=100,max_steps=200,futr_exog_list= futr_exog_l)
    fcst = NeuralForecast(models=[model], freq=frequency)
    fcst.fit(df=hist_df)
    drnnforecasts = fcst.predict(futr_df=futr_dfx)
    return drnnforecasts

def GRUforecaster(dataset: pd.DataFrame,hor:int,a:str):
    featuredataset = dataset.drop(['ds','y','unique_id'],axis = 1)
    featurelist = featuredataset.columns.to_list()
    exog_fut_data = []
    for feature in featurelist:
        dx = dataset[['ds',feature]]
        frequency = pd.infer_freq(dataset['ds'])
        dx.set_index('ds',inplace=True)        
        fitted_model = ExponentialSmoothing(dx[feature],trend = 'additive',seasonal= 'additive',\
                                            seasonal_periods=12,freq= frequency).fit()
        dz = fitted_model.forecast(hor)
        dz = dz.to_frame()
        dz = dz.reset_index()
        dz = dz.rename(columns= {'index': 'ds',0: feature})
        exog_fut_data.append(dz)
    exog = [df.set_index('ds') for df in exog_fut_data]
    exog_future = pd.concat(exog, axis=1)
    exog_future = exog_future.reset_index()
    exog_future.insert(0, "unique_id", a) 
    futr_dfx = exog_future.reset_index()
    static_dfx = dataset['unique_id'].to_frame()
    #dataset = dataset.rename(columns={'Index':'count'})
    dataset['ds'] = pd.to_datetime(dataset['ds'])
    hist_df = dataset#.drop('count',axis = 1)
    hist_df['ds'] = pd.to_datetime(hist_df['ds'])
    futr_dfx = futr_dfx.rename(columns={'index':'count'})
    futr_dfx = futr_dfx.drop('count',axis = 1)
    futr_exog_l =   exog_future.drop(['unique_id','ds'],axis=1).columns.to_list()                   
    model = GRU(h=12,input_size=-1,loss=DistributionLoss(distribution='Normal', level=[80, 90]),scaler_type='robust',encoder_n_layers=2,encoder_hidden_size=128,context_size=10,decoder_hidden_size=128,decoder_layers=2,max_steps=200,futr_exog_list=futr_exog_l )
    fcst = NeuralForecast(models=[model], freq=frequency)
    fcst.fit(df=hist_df)
    gruforecasts = fcst.predict(futr_df=futr_dfx)
    return gruforecasts

def LSTMforecaster(dataset: pd.DataFrame,hor:int,a:str):
    featuredataset = dataset.drop(['ds','y','unique_id'],axis = 1)
    featurelist = featuredataset.columns.to_list()
    exog_fut_data = []
    for feature in featurelist:
        dx = dataset[['ds',feature]]
        frequency = pd.infer_freq(dataset['ds'])
        dx.set_index('ds',inplace=True)        
        fitted_model = ExponentialSmoothing(dx[feature],trend = 'additive',seasonal= 'additive',seasonal_periods=12,freq= frequency).fit()
        dz = fitted_model.forecast(hor)
        dz = dz.to_frame()
        dz = dz.reset_index()
        dz = dz.rename(columns= {'index': 'ds',0: feature})
        exog_fut_data.append(dz)
    exog = [df.set_index('ds') for df in exog_fut_data]
    exog_future = pd.concat(exog, axis=1)
    exog_future = exog_future.reset_index()
    exog_future.insert(0, "unique_id", a) 
    futr_dfx = exog_future.reset_index()
    static_dfx = dataset['unique_id'].to_frame()
    #dataset = dataset.rename(columns={'Index':'count'})
    dataset['ds'] = pd.to_datetime(dataset['ds'])
    hist_df = dataset#.drop('count',axis = 1)
    hist_df['ds'] = pd.to_datetime(hist_df['ds'])
    futr_dfx = futr_dfx.rename(columns={'index':'count'})
    futr_dfx = futr_dfx.drop('count',axis = 1)
    futr_exog_l =   exog_future.drop(['unique_id','ds'],axis=1).columns.to_list()                   
    model = LSTM(h=12,input_size=-1,loss=DistributionLoss(distribution='Normal', level=[80, 90]),scaler_type='robust',\
                 encoder_n_layers=2,encoder_hidden_size=128,context_size=10,decoder_hidden_size=128,decoder_layers=2,max_steps=200,futr_exog_list=futr_exog_l )
    fcst = NeuralForecast(models=[model], freq=frequency)
    fcst.fit(df=hist_df)
    lstmforecasts = fcst.predict(futr_df=futr_dfx)
    return lstmforecasts


def DeepARforecaster(dataset: pd.DataFrame,hor:int,a:str):
    featuredataset = dataset.drop(['ds','y','unique_id'],axis = 1)
    featurelist = featuredataset.columns.to_list()
    exog_fut_data = []
    for feature in featurelist:
        dx = dataset[['ds',feature]]
        frequency = pd.infer_freq(dataset['ds'])
        dx.set_index('ds',inplace=True)        
        fitted_model = ExponentialSmoothing(dx[feature],trend = 'additive',seasonal= 'additive',seasonal_periods=12,freq= frequency).fit()
        dz = fitted_model.forecast(hor)
        dz = dz.to_frame()
        dz = dz.reset_index()
        dz = dz.rename(columns= {'index': 'ds',0: feature})
        exog_fut_data.append(dz)
    exog = [df.set_index('ds') for df in exog_fut_data]
    exog_future = pd.concat(exog, axis=1)
    exog_future = exog_future.reset_index()
    exog_future.insert(0, "unique_id", a) 
    futr_dfx = exog_future.reset_index()
    static_dfx = dataset['unique_id'].to_frame()
    #dataset = dataset.rename(columns={'Index':'count'})
    dataset['ds'] = pd.to_datetime(dataset['ds'])
    hist_df = dataset#.drop('count',axis = 1)
    hist_df['ds'] = pd.to_datetime(hist_df['ds'])
    futr_dfx = futr_dfx.rename(columns={'index':'count'})
    futr_dfx = futr_dfx.drop('count',axis = 1)
    futr_exog_l =   exog_future.drop(['unique_id','ds'],axis=1).columns.to_list()                   
    model = DeepAR(h=12,input_size=48,lstm_n_layers=3,trajectory_samples=100,loss=DistributionLoss(distribution='Normal', level=[80, 90], \
                    return_params=False),learning_rate=0.005,futr_exog_list=futr_exog_l,max_steps=100,\
                        val_check_steps=10,early_stop_patience_steps=-1,scaler_type='standard',enable_progress_bar=True)
    fcst = NeuralForecast(models=[model], freq=frequency)
    fcst.fit(df=hist_df)
    deeparforecasts = fcst.predict(futr_df=futr_dfx)
    return deeparforecasts