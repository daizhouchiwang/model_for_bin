import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


"""
Define model struture
"""
def define_model_structure():
    var_base = {'T_spline_6_8': ("+ bs(tave6, df=3, knots = (20,23), degree=1,lower_bound=7,upper_bound=29)"
                            + "+ bs(tave7, df=3, knots = (22,26), degree=1,lower_bound=10,upper_bound=32)" 
                            + "+ bs(tave8, df=3, knots = (20,24), degree=1,lower_bound=11,upper_bound=31)"), 
                                
            'P_spline_6_9': (" + bs(precip6, df=3, knots = (75,200), degree=1,lower_bound=0,upper_bound=700)"
                            + " + bs(precip7, df=3, knots = (75,200), degree=1,lower_bound=0,upper_bound=700)"
                            + " + bs(precip8, df=2, knots = (90,), degree=1,lower_bound=0,upper_bound=700)"
                            + " + bs(precip9, df=3, knots = (100,200), degree=1, lower_bound=0, upper_bound=750)"),
           }
    
    model_vars = var_base['T_spline_6_8'] + var_base['P_spline_6_9']
        
    return ("Q('%s_ana') ~ "%'yield_rainfed' + model_vars + "+ C(FIPS)")

"""
Estimate global yearly linear trend
"""
def yield_trend(df):
    trend_model_txt = "yield_rainfed ~ year"
    trend_results = smf.ols(trend_model_txt, data=df).fit()
    return trend_results


"""
Load the obs yield data for model training 
"""
def load_yield_data():    
    # Load train_data
    data = pd.read_csv('./Corn_model_data.csv',dtype={'FIPS':str})

    data['corn_percent'] = data['area']/data['land_area']

    # Add logical filter to the yield Data
    area_con = data['area'].notnull()
    data = data[area_con]

    # Add Rainfed yield
    # rainfed_con: counties without irrigation, the yield is rainfed
    rainfed_con = ~data['FIPS'].isin(data.loc[data['yield_irr'].notnull(),'FIPS'].unique())
    data['yield_rainfed'] = data['yield_noirr']
    data['area_rainfed'] = data['area_noirr']

    # For counties with irrigation, only the rainfed yield is added to irrigated yield
    data.loc[rainfed_con, 'yield_rainfed'] = data.loc[rainfed_con, 'yield']
    data.loc[rainfed_con, 'area_rainfed'] = data.loc[rainfed_con, 'area']


    # 12 core states
    states_12 = ['ILLINOIS', 'INDIANA', 'IOWA', 'KANSAS', 'MICHIGAN', 'MINNESOTA',
                 'MISSOURI', 'NEBRASKA', 'NORTH DAKOTA', 'OHIO', 'SOUTH DAKOTA',
                 'WISCONSIN']

    data_12 = data[data['State'].isin(states_12)]
    
    return data_12

"""
Train the model and save model to local file
"""
def train_model():
    train_data = load_yield_data()

    # Detrend rainfed yield
    trend_rainfed = yield_trend(train_data)
    train_data.loc[:,'yield_rainfed_ana'] = (train_data['yield_rainfed'] - trend_rainfed.predict(train_data[['year','yield_rainfed']]))      

    model_txt = define_model_structure()
    
    model = smf.ols(model_txt, data=train_data, missing='drop').fit()
    
    print('The trained model is saved at yield_tave_precip_spline_model.pickle')
    model.save("yield_tave_precip_spline_model.pickle")
    
    print('The yield trend model is saved at yield_trend_rainfed_model.pickle')
    trend_rainfed.save("yield_trend_rainfed_model.pickle")
    


"""
Basic function to predict yield 
df_predict = make_prediction(df)
"""
def make_prediction(df):
    # Load model 
    trained_model = sm.load('yield_tave_precip_spline_model.pickle')
    trend_results = sm.load('yield_trend_rainfed_model.pickle')
    # Get predicted yield and attach it to existing dataframe
    df_predict = df.copy().join(trained_model.predict(df).to_frame('predicted_yield_rainfed_ana'))
    # Add trend term to get yield
    df_predict['predicted_yield_rainfed'] = df_predict['predicted_yield_rainfed_ana'] + trend_results.predict(df_predict['year'])
    return df_predict


"""
Load observed climate data
"""
def load_obs_climate():
    obs_climate = pd.read_csv('./prism_climate_growing_season_1981_2016.csv',dtype={'FIPS':str})
    return obs_climate.set_index(['year','FIPS'])



"""
Load forecast data
df = load_forecast_data(year, month, ens)
E.g., d = load_forecast_data(1982, 1, 1)
"""
def load_forecast_data(year, month, ens):
    d = pd.read_csv('./NCEP_CFS_forcing/NCEP_CFS_%d%02d_ens%02d.csv'%(year,month,ens),dtype={'FIPS':str})
    # convert to monthly total amount
    d.loc[:,'precip5'] = d.loc[:,'precip5'] * 31
    d.loc[:,'precip6'] = d.loc[:,'precip6'] * 30
    d.loc[:,'precip7'] = d.loc[:,'precip7'] * 31
    d.loc[:,'precip8'] = d.loc[:,'precip8'] * 31
    d.loc[:,'precip9'] = d.loc[:,'precip9'] * 30

    d.rename(columns={'tavg5':'tave5','tavg6':'tave6','tavg7':'tave7','tavg8':'tave8','tavg9':'tave9'},inplace=True)
    
    # Convert to Degree C
    d.loc[:,'tave5':'tave9'] = d.loc[:,'tave5':'tave9'] - 273.15 

    # Add year column
    d['year'] = year
    
    return d


"""
Merge forecast with observed climate at the time of a month for prediction
Prediction will be made at 1st of that month
e.g., month = 7, means prediction and 1st July, thus replace June and May with observed climate

d = merge_forecast_with_obs(d_forecast, d_obs, month)
"""
# It works for both a single year or multiple years
def merge_forecast_with_obs(d_forecast, d_obs, month):
    
    d_forecast.set_index(['year','FIPS'],inplace=True)
    
    # month number that needs to be filled with obs, starting from June
    n = month - 6 + 1
    
    # if month >=7, replace month - 1 with obs
    if n > 0:
        for i in range(1,n+1):
            d_forecast.loc[:,'precip' + str(month-i)] = d_obs.loc[:,'precip' + str(month-i)]
            d_forecast.loc[:,'tave' + str(month-i)] = d_obs.loc[:,'tave' + str(month-i)]
            
    return d_forecast.reset_index()  


"""
Predict yield for multiple years at once and return the results
e.g., d_final = get_prediction_result(1982, 1990, 1, 1)
"""
def get_prediction_result(start_year, end_year, prediction_month, ens_number):
    # Load forecast data for multiple years
    d_forecast = pd.concat([load_forecast_data(y, prediction_month, ens_number) for y in range(start_year,end_year)])
    
    # Merge forecast with obs climate depending on the prediction month
    if prediction_month>=6:
        obs_climate = load_obs_climate()
        d_forecast = merge_forecast_with_obs(d_forecast, obs_climate, prediction_month)
    
    # Load obs yield data
    obs_yield = load_yield_data()
    
    # Only predict counties that included in the 12 states obs data 
    con = d_forecast['FIPS'].isin(obs_yield['FIPS'].unique())
    dp = make_prediction(d_forecast[con].reset_index())
    
    # Combine with obs yield data
    dp_final = dp.loc[:,['year','FIPS','predicted_yield_rainfed']].merge(obs_yield[['year','FIPS','yield_rainfed']])

    dp_final.to_csv('./result/yield_forcast_%d_%d_at_month_%02d_ens%02d.csv'%(start_year,end_year,
                                                                          prediction_month, ens_number))
    print('Yield prediction for %d_%d_at_month_%02d_ens%02d saved to file'%(start_year,end_year,
                                                                          prediction_month, ens_number))
    return dp_final

if __name__ == "__main__":
# first use train_model() to train the model and save the model to local file
#    
 
# Below save prediction results for multiple years and ensembles
#    ens_number=1
#    prediction_month = 5
    start_year = 1982
    end_year = 1990

    for m in range(1,10):
        for e in range(1,25):
            get_prediction_result(start_year, end_year, m, e)



