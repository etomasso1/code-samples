import requests
import pandas as pd
import numpy as np
import json
import os
import pickle
import copy

from datetime import datetime, timedelta, date, time
from catboost import CatBoostRegressor

def clean_teetimes(df_t):
    df = df_t.copy()
    ########################################################################################
    # Drop duplicates .loc[:, ('one', 'second')]
    ########################################################################################
    df.drop_duplicates(subset=["date", "time", "golfcourse_id"], inplace=True)
    ########################################################################################

    # Convert dts column (time) to datetime
    df['time']= pd.to_datetime(df.loc[:,'time'])
    # Add a column for hour
    df['hour']= df['time'].dt.hour

    df['time'] = df['time'].dt.time
    #df['time'] = df.loc[:,'time'].astype(str).str[:-3]

    # Convert date to datetime
    df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'])

    # Create a new column with the date days of the week
    df['date_dow'] = df['date'].dt.dayofweek

    df['date_weekday'] = df['date'].dt.day_name

    # Add a column to check if the teetime date is a weekend
    df['date_weekend'] = [(1 if (t == 5 or t ==6) else 0) for t in df.loc[:,'date_dow']]

    # Convert timestamp column to datetime
    df['timestamp']= pd.to_datetime(df['timestamp'])

    # Create a new column that shows the timespan (timedelta) between the time the data was taken
    df['span_from'] = df['date'] - df['timestamp']

    # Convert date to datetime.date
    df['date'] = df['date'].dt.date

    # Convert that timespan into days
    df['days_from'] = df['span_from'].dt.days

    # Create a new column for timestamp days of the week
    df['timestamp_dow'] = df['timestamp'].dt.dayofweek

    # Add a column to see if the timestamps were taken on a weekend
    df['timestamp_weekend'] = [(1 if (t == 5 or t ==6) else 0) for t in df.loc[:,'timestamp_dow']]

    # Get the hour of each timestamp
    df['timestamp_hour'] = df['timestamp'].dt.hour

    return df



def get_preds():
    today = date.today()
    # Production server URL
    request_url = ######
    payload = json.dumps({'code': '#####'})

    response = requests.get(request_url, data=payload)

    df_t = pd.DataFrame(response.json(), columns=['id', 'golfcourse_id', 'course_name', 'time',
                                                     'date', 'timestamp', 'hour', 'week', 'month', 'year', 'times_updated'])

    df_t = clean_teetimes(df_t)
    df_t['time'] = df_t['time'].astype(str).str[:-3]

    # Weather Forecast Data
    ####################################################################################################

    # Production server URL
    request_url = '######'

    # Demo course
    #payload = json.dumps({'code': '#######', 'course': '#####'})

    # All courses
    payload = json.dumps({'code': '#######'})

    wf = requests.get(request_url, data=payload)
    jwf = pd.DataFrame(wf.json())


    # Weather Forecast Data Transformations
    ####################################################################################################
    # Forecast data to make predictions

    # Convert date of weather forecast to datetime
    jwf.loc[:, 'date'] = pd.to_datetime(jwf.loc[:, 'date'], utc=True)

    # Convert timestamp to datetime for when the weather forecast was taken
    jwf.loc[:, 'timestamp'] = pd.to_datetime(jwf.loc[:, 'timestamp'])

    # Get the days_from by subtracting forecast day from timestamp, then adding 1
    jwf['days_from'] = (jwf['date'] - jwf['timestamp']).dt.days + 1

    # Convert date to datetime.date
    jwf.loc[:, 'date'] = jwf['date'].dt.date

    # Boolean mask for date range
    mask = df_t['date'] > date(2020, 9, 15)
    df_tf = df_t.loc[mask].copy(deep=True)

    # Drop duplicates from the teetime dataset
    df_tf.drop_duplicates(subset=["date", "time", "golfcourse_id"], inplace=True)

    # Merge dataframes based on date, golfcourse_id
    df_t_jwf = pd.merge(df_tf, jwf, how="left", on=['date', 'golfcourse_id'], indicator=True)

    # Only keep the rows that show up in both dataframes
    df_t_jwf = df_t_jwf.loc[df_t_jwf['_merge'] == "both"]
    df_t_jwf.drop(columns=['_merge', 'id_x', 'id_y', "times_updated", 'course_name',
                            'timestamp_x', 'timestamp_y', 'span_from', 'date_weekday'], inplace=True)

    df_t_jwf.drop_duplicates(subset=["date", "time", "golfcourse_id"], inplace=True)

    save_the_date = df_t_jwf['date'].copy(deep=True)
    save_the_time = df_t_jwf['time'].copy(deep=True)
    save_the_days = df_t_jwf['days_from_y'].copy(deep=True)
    save_the_gcid = df_t_jwf['golfcourse_id'].copy(deep=True)

    df_output = df_t_jwf[['date', 'time', 'days_from_y', 'golfcourse_id']]
    mask_output = (df_output['date'] > today)
    df_output = df_output.loc[mask_output]

    #######################################################################################################
    # One-hot encoding of cateogorical variables
    #######################################################################################################
    # Define the categorical columns to convert
    cols_to_convert = ['golfcourse_id', 'time', 'date', 'hour', 'week', 'month', 'year', 'date_dow','date_weekend',
                          'days_from_x', 'days_from_y', 'timestamp_dow', 'timestamp_weekend', 'timestamp_hour',
                       'condition', 'zip_code']

    df_t_jwf[cols_to_convert] = df_t_jwf[cols_to_convert].astype('object')
    vars_ind_categorical = df_t_jwf.select_dtypes(include=['object']).columns.tolist()

    vars_all = df_t_jwf.columns.values


    vars_ind_onehot = []

    df_all_onehot = df_t_jwf.copy(deep=True)

    for col in vars_ind_categorical:

        # use pd.get_dummies on  df_all[col]
        df_oh = pd.get_dummies(df_t_jwf[col], drop_first=False)
        # Rename the columns to have the original variable name as a prefix
        oh_names = [col + '_' + str(c) for c in df_oh.columns]
        #col + '_' + df_oh.columns
        df_oh.columns = oh_names

        df_all_onehot = pd.concat([df_all_onehot, df_oh], axis = 1, sort = False)

        del df_all_onehot[col]
        vars_ind_onehot.extend(oh_names)

    #rng = np.random.RandomState(2020)
    #fold = rng.randint(0, 10, df_all_onehot.shape[0])
    #df_all_onehot['fold'] = fold

    # rename df_all_onehot to df_all as this is now the data we will be using for
    # the rest of this work
    df_t_jwf = df_all_onehot.copy(deep=True)

    del df_all_onehot

    vars_ind = df_t_jwf.columns.tolist()

    #######################################################################################################
    # Copy dataframes and split into train and prediction sets
    #######################################################################################################

    df_t_jwf['date'] = save_the_date
    df_t_jwf['golfcourse_id'] = save_the_gcid

    df_train = df_t_jwf.copy(deep=True)
    df_predict = df_t_jwf.copy(deep=True)


    # Boolean mask for date range
    mask_train = (df_train['date'] > date(2020, 9, 15)) & (df_train['date'] <= today)
    df_train = df_train.loc[mask_train].copy(deep=True)

    ###########################################################################################################
    # Our dependent variable
    ###########################################################################################################
    var_dep = 'counts_daily'

    # Get the aggregate number of teetimes for each day
    date_course_agg = df_train.groupby(['date', 'golfcourse_id']).size().reset_index()
    # Rename 3rd column to counts_daily
    date_course_agg.rename(columns = { date_course_agg.columns[2]: "counts_daily" }, inplace=True)

    # Merge dataframes based on date, golfcourse_id
    df_train = pd.merge(df_train, date_course_agg, on=['date', 'golfcourse_id'])


    # Drop the golfcourse_id and date columns
    df_train.drop(columns=['date', 'golfcourse_id'], inplace=True)

    vars_ind_train = df_train.columns.tolist()
    vars_ind_train.remove(var_dep)

    ###########################################################################################################
    # Set up train/test splits
    ###########################################################################################################
    rng = np.random.RandomState(2020)
    fold = rng.randint(0, 10, df_train.shape[0])
    df_train['fold'] = fold

    # define index for train, val, design, test
    idx_train  = np.where(df_train['fold'].isin(np.arange(0,6)))[0]
    idx_val    = np.where(df_train['fold'].isin([6,7]))[0]
    idx_design = np.where(df_train['fold'].isin(np.arange(0,8)))[0]
    idx_test   = np.where(df_train['fold'].isin([8,9]))[0]

    X = df_train[vars_ind].values
    y = df_train[var_dep].values

    X_train  = X[idx_train, :]
    X_val    = X[idx_val, :]
    X_design = X[idx_design, :]
    X_test   = X[idx_test, :]

    y_train  = df_train[var_dep].iloc[idx_train].copy().values.ravel()
    y_val    = df_train[var_dep].iloc[idx_val].copy().values.ravel()
    y_design = df_train[var_dep].iloc[idx_design].copy().values.ravel()
    y_test   = df_train[var_dep].iloc[idx_test].copy().values.ravel()

    X_train = X_train.astype(int)
    y_train = y_train.astype(int)
    X_val = X_val.astype(int)
    y_val = y_val.astype(int)
    X_design = X_design.astype(int)
    y_design = y_design.astype(int)
    X_test = X_test.astype(int)
    y_test = y_test.astype(int)

    ###########################################################################################################
    # Model Train
    ###########################################################################################################
    model_regress = CatBoostRegressor(iterations=4000,
                          use_best_model=True,
                          eval_metric='R2',
                          od_type='Iter',
                          od_wait=1000,
                          boosting_type='Plain',
                          bootstrap_type='Bernoulli',
                          one_hot_max_size=10,
                          #By default, the learning rate is defined automatically
                          #based on the dataset properties and the number of
                          #iterations. The automatically defined value should
                          #be close to the optimal one.
                          #learning_rate=0.002,
                          depth=6,
                          l2_leaf_reg=3, # default is 3
                          loss_function='RMSE',
                          random_seed=2020,
                          logging_level='Silent'
                          )

    model_regress.fit(X_train, y_train,
              eval_set=(X_val, y_val),
              logging_level='Silent',
              plot=False)

    ###########################################################################################################
    # Save model and get ready for predictions
    ###########################################################################################################

    os_models = "##########"
    model_regress.save_model(os.path.join(os_models, 'cb1_wf'))


    # Forecast data to make predictions - df_predict
    df_predict['date'] = save_the_date
    # Boolean mask for date range
    df_mask = (df_predict['date'] > today)
    df_predict = df_predict.loc[df_mask]

    df_predict.drop(columns=['date'], inplace=True)
    vars_predict = df_predict.columns.tolist()
    ###########################################################################################################
    # Predictions
    ###########################################################################################################
    X_predict = df_predict[vars_predict].values

    predictions = model_regress.predict(X_predict)


    df_output['preds'] = predictions.round(0)
    df_preds = df_output.groupby(['date', 'golfcourse_id'])['preds'].mean().round(0).reset_index()


    os_results = "######"
    fname = 'df_preds-' + str(date.today()) + '.pkl'

    df_preds.to_pickle(os.path.join(os_results, fname))
    return df_preds
