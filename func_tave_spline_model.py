import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


global yield_type_dict
yield_type_dict = {'all': 'yield', 'rainfed':'yield_rainfed','irrigated':'yield_irr'}

"""
Define model struture
"""
def define_model_structure(model_name, yield_type='rainfed'):
    var_base = {
                'T_spline_6_8': ("+ bs(tave6, df=3, knots = (20,23), degree=1,lower_bound=7,upper_bound=29)"
                         + "+ bs(tave7, df=3, knots = (22,26), degree=1,lower_bound=10,upper_bound=32)" 
                         + "+ bs(tave8, df=3, knots = (20,24), degree=1,lower_bound=11,upper_bound=31)"), 

             'T_spline_6': "+ bs(tave6, df=3, knots = (20,23), degree=1,lower_bound=7,upper_bound=29)",
             'T_spline_7': "+ bs(tave7, df=3, knots = (22,26), degree=1,lower_bound=10,upper_bound=32)", 
             'T_spline_8': "+ bs(tave8, df=3, knots = (20,24), degree=1,lower_bound=11,upper_bound=31)",

           # 'VPD_spline_6_8': ("+ bs(vpdave6, df=5, knots = (8,10,13,15), degree=1,lower_bound=4,upper_bound=30)"
           #                   + "+ bs(vpdave7, df=3, knots = (8,11), degree=1,lower_bound=4,upper_bound=35)"
           #                   + " + bs(vpdave8, df=3, knots = (8,15), degree=1,lower_bound=3,upper_bound=30)"),

             'VPD_spline_6': "+ bs(vpdave6, df=5, knots = (8,10,13,15), degree=1,lower_bound=4,upper_bound=30)",
             'VPD_spline_7': "+ bs(vpdave7, df=3, knots = (8,11), degree=1,lower_bound=4,upper_bound=35)",
             'VPD_spline_8': "+ bs(vpdave8, df=3, knots = (8,15), degree=1,lower_bound=3,upper_bound=30)",

                                
            'P_spline_6_9': (" + bs(precip6, df=3, knots = (75,200), degree=1,lower_bound=0,upper_bound=700)"
                            + " + bs(precip7, df=3, knots = (75,200), degree=1,lower_bound=0,upper_bound=700)"
                            + " + bs(precip8, df=2, knots = (90,), degree=1,lower_bound=0,upper_bound=700)"
                            + " + bs(precip9, df=3, knots = (100,200), degree=1, lower_bound=0, upper_bound=750)"),

            'EVI_poly_5': "+ evi5 + np.power(evi5,2)",
            'EVI_poly_6': "+ evi6 + np.power(evi6,2)",
            'EVI_poly_7': "+ evi7 + np.power(evi7,2)",
            'EVI_poly_8': "+ evi8 + np.power(evi8,2)"

           }
    
    if model_name == 'tave_full':
        model_vars = var_base['T_spline_6_8'] + var_base['P_spline_6_9']

    if model_name == 'tave_full_vpd_july':
        model_vars = var_base['T_spline_6_8'] + var_base['P_spline_6_9'] + var_base['VPD_spline_6']

    if model_name == 'tave_full_vpd_aug':
        model_vars = var_base['T_spline_6_8'] + var_base['P_spline_6_9'] + var_base['VPD_spline_6'] \
                      + var_base['VPD_spline_7']

    if model_name == 'tave_full_vpd_sep':
        model_vars = var_base['T_spline_6_8'] + var_base['P_spline_6_9'] + var_base['VPD_spline_6'] \
                      + var_base['VPD_spline_7'] + var_base['VPD_spline_8']

    if model_name == 'tave_full_evi_june':
        model_vars = var_base['T_spline_6_8'] + var_base['P_spline_6_9'] + var_base['EVI_poly_5']

    if model_name == 'tave_full_evi_july':
        model_vars = var_base['T_spline_6_8'] + var_base['P_spline_6_9'] + var_base['EVI_poly_5'] \
                     + var_base['EVI_poly_6']

    if model_name == 'tave_full_evi_aug':
        model_vars = var_base['T_spline_6_8'] + var_base['P_spline_6_9'] + var_base['EVI_poly_5'] \
                     + var_base['EVI_poly_6'] + var_base['EVI_poly_7']

    if model_name == 'tave_full_evi_sep':
        model_vars = var_base['T_spline_6_8'] + var_base['P_spline_6_9'] + var_base['EVI_poly_5'] \
                     + var_base['EVI_poly_6'] + var_base['EVI_poly_7'] + var_base['EVI_poly_8']
        
    return ("Q('%s_ana') ~ "%yield_type_dict[yield_type] + model_vars + "+ C(FIPS)")


"""
Estimate the global yield trend
"""
def yield_trend(df, yield_type='rainfed'):
    trend_model_txt = "Q('%s')"%yield_type_dict[yield_type] + "~ year"
    trend_results = smf.ols(trend_model_txt, data=df).fit()
    return trend_results

"""
Save rainfed and irrigated yield trend 
"""
def save_yield_trend():
    train_data = load_yield_data()
    trend_rainfed = yield_trend(train_data)
    trend_irrigated = yield_trend(train_data,yield_type='irrigated')

    trend_rainfed.save("yield_trend_rainfed_model.pickle")
    print('The yield trend model saved as yield_trend_rainfed_model.pickle')

    trend_irrigated.save("yield_trend_irrigated_model.pickle")
    print('The yield trend model saved as yield_trend_irrigated_model.pickle')


"""
Load saved yield trend
"""
def load_yield_trend(yield_type):
    trend_model = sm.load("yield_trend_%s_model.pickle"%yield_type)
    return trend_model 

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
def train_model(model_name,yield_type='rainfed'):
    train_data = load_yield_data()

    # Detrend rainfed yield
    trend_rainfed = load_yield_trend(yield_type)

    train_data.loc[:,yield_type_dict[yield_type]+'_ana'] = (train_data[yield_type_dict[yield_type]] - 
                                             trend_rainfed.predict(train_data[['year',yield_type_dict[yield_type]]]))

    model_txt = define_model_structure(model_name,yield_type=yield_type)
    
    model = smf.ols(model_txt, data=train_data, missing='drop').fit()
    
    print('The trained model saved at yield_%s_model.pickle'%model_name)
    model.save("yield_%s_model.pickle"%model_name)
    



"""
Basic function to predict yield 
df_predict = make_prediction(df)
"""
def make_prediction(df, model_name,yield_type='rainfed'):
    # Load model 
    trained_model = sm.load('yield_%s_model.pickle'%model_name)

    trend_results = load_yield_trend(yield_type)

    # Get predicted yield and attach it to existing dataframe
    df_predict = df.copy().join(trained_model.predict(df).to_frame('predicted_yield_%s_ana'%yield_type))
    # Add trend term to get yield
    df_predict['predicted_yield_%s'%yield_type] = df_predict['predicted_yield_%s_ana'%yield_type] \
                                + trend_results.predict(df_predict['year'])
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

    # Append vpd data        
#    d_forecast = d_forecast.join(d_obs.loc[:,'vpdave6':'vpdave8'])

#    return d_forecast.reset_index()  
    return d_forecast.join(d_obs.loc[:,'vpdave6':'vpdave8']).reset_index()  


"""
Predict yield for multiple years at once and return the results
e.g., d_final = get_prediction_result(1982, 1990, 1, 1)
"""
def get_prediction_result(start_year, end_year, prediction_month, ens_number,model_name='tave_full',yield_type='rainfed'):
    # Load forecast data for multiple years
    d_forecast = pd.concat([load_forecast_data(y, prediction_month, ens_number) for y in range(start_year,end_year)])
    
    # Merge forecast with obs climate depending on the prediction month
    if prediction_month>=6:
        obs_climate = load_obs_climate()
        d_forecast = merge_forecast_with_obs(d_forecast, obs_climate, prediction_month)
    print(prediction_month)
    print(d_forecast.columns)

    # Append vpd data        
#    d_forecast = d_forecast.merge(obs_climate.loc[:,'vpdave6':'vpdave8'].reset_index(),on=['year','FIPS'])
    
    # Load obs yield data
    obs_yield = load_yield_data()
    
    # Only predict counties that included in the 12 states obs data 
    con = d_forecast['FIPS'].isin(obs_yield['FIPS'].unique())
    dp = make_prediction(d_forecast[con].reset_index(),model_name,yield_type='rainfed')
    
    # Combine with obs yield data
    dp_final = dp.loc[:,['year','FIPS','predicted_yield_'+yield_type]].merge(obs_yield[['year','FIPS',yield_type_dict[yield_type]]])

    dp_final.to_csv('./result/%s_yield_%s_forcast_%d_%d_at_month_%02d_ens%02d.csv'%(model_name,yield_type,start_year,end_year,
                                                                          prediction_month, ens_number))
    print('Prediction for %s with model %s %d_%d_at_month_%02d_ens%02d saved to file'%(yield_type,model_name,start_year,end_year,
                                                                          prediction_month, ens_number))
    return dp_final

def train_model_batch():
    models = ['tave_full','tave_full_vpd_july','tave_full_vpd_aug','tave_full_vpd_sep',
              'tave_full_evi_june','tave_full_evi_july','tave_full_evi_aug','tave_full_evi_sep']
    yield_types = ['rainfed','irrigated']
    for y in yield_types:
        for m in models:
            train_model(m,yield_type=y)

if __name__ == "__main__":
## Save yield trend to local file
#    save_yield_trend()

# train_model_batch() to train many models and save each of them to local file
#    train_model_batch()
 
# Below save prediction results for multiple years and ensembles
# Note that models with vpd can only be used when prediction month starting from June
    ens_number=1
    prediction_month = 6
    start_year = 1982
    end_year = 1984

    for m in range(6,10):
        for e in range(1,25):
            get_prediction_result(start_year, end_year, m, e, model_name='tave_full_vpd_july',yield_type='rainfed')

