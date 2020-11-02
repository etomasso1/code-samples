from datetime import date

import pandas as pd
import copy

from .data_manipulation import (CategoricalVariables, clean_and_aggregate,
                                RemoteRequests, merge_teetimes_forecast)
from .create_ml_model import MLModel


def get_preds():
    """Get predictions."""
    # Call teetime static method to clean and return dataframe
    df_t = RemoteRequests.teetime_data(
                            request_url='****'
                            )
    # Call weather forecast static method to clean and return dataframe
    jwf = RemoteRequests.weather_forecast(
                            request_url='****'
                            )

    save_the_date, save_the_gcid, df_output, df_t_jwf \
        = merge_teetimes_forecast(df_t, jwf)

    ###########################################################################
    # One-hot encoding of cateogorical variables
    ###########################################################################
    cat_output = CategoricalVariables(df_t_jwf)
    df_t_jwf = cat_output.categorical_dataframe()
    vars_ind = cat_output.independent_vars()

    ###########################################################################
    # Copy dataframes and split into train and prediction sets
    ###########################################################################
    df_t_jwf['date'] = save_the_date
    df_t_jwf['golfcourse_id'] = save_the_gcid

    df_train = clean_and_aggregate(df_t_jwf.copy(deep=True))
    df_predict = df_t_jwf.copy(deep=True)

    ###########################################################################
    # Train ML Model and return results
    ###########################################################################
    df_preds = MLModel(df_train, df_predict, df_output,
                       vars_ind, save_the_date).return_df_preds()

    return df_preds
