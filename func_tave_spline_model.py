import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


"""
Define model struture
"""
def define_model_structure():
    var_base = {'T_spline_6_8': ("+ bs(tave6, df=3, knots = (20,23), degree=1,lower_bound=12,upper_bound=29)"
                            + "+ bs(tave7, df=3, knots = (22,26), degree=1,lower_bound=15,upper_bound=32)" 
                            + "+ bs(tave8, df=3, knots = (20,24), degree=1,lower_bound=14,upper_bound=31)"), 
                                
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
Make yield 
df_predict = make_prediction(df)
"""
def make_prediction(df):
    # Load model 
    trained_model = sm.load('yield_tave_precip_spline_model.pickle')
    trend_results = sm.load('yield_trend_rainfed_model.pickle')
    # Get predicted yield and attach it to existing dataframe
    df_predict = df.copy().join(trained_model.predict(df).to_frame('predicted_yield_rainfed_ana'))
    # Add trend term to get yield
    df_predict['predicted_rainfed_yield'] = df_predict['predicted_yield_rainfed_ana'] + trend_results.predict(df_predict['year'])
    return df_predict


# In[26]:

# Step 1: train the model, save the model to local file (optional)
# train_model()

# Step 2: make prediction: need data frame "d" with columns: 'year','FIPS','tave6':'tave8', and 'precip6':'precip9'

# Example:
# d = load_yield_data()
# df_predict = make_prediction(d)

