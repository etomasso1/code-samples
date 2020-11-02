import json
from datetime import date

import pandas as pd

import requests


def clean_teetimes(df_t):
    """Clean teetime data, convert to datetime."""
    df = df_t.copy()
    ###########################################################################
    # Drop duplicates .loc[:, ('one', 'second')]
    ###########################################################################
    DataConversions.drop_duplicates(df, ["date", "time", "golfcourse_id"])
    ###########################################################################
    # Convert column time to datetime
    DataConversions.pd_datetime(df, 'time')

    # Add a column for hour
    df = DataConversions.dt_hour(df, 'hour', 'time')
    # Convert column to datetime
    df = DataConversions.dt_time(df, 'time')
    # Convert date to datetime
    df = DataConversions.pd_datetime(df, 'date')
    # Create a new column with the date days of the week
    df = DataConversions.dt_dayofweek(df, 'date_dow', 'date')
    # df['date_weekday'] = df['date'].dt.day_name
    df = DataConversions.dt_day_name(df, 'date_weekday', 'date')
    # Add a column to check if the teetime date is a weekend
    df = DataConversions.if_weekend(df, 'date_weekend', 'date_dow')
    # Convert timestamp column to datetime
    df = DataConversions.pd_datetime(df, 'timestamp')
    # Create a new column that shows the timespan
    df['span_from'] = df['date'] - df['timestamp']
    # Convert date to datetime.date
    df = DataConversions.dt_date(df, 'date')
    # Convert that timespan into days
    df = DataConversions.dt_days(df, 'days_from', 'span_from')
    # Create a new column for timestamp days of the week
    df = DataConversions.dt_dayofweek(df, 'timestamp_dow', 'timestamp')
    # Add a column to see if the timestamps were taken on a weekend
    df = DataConversions.if_weekend(df, 'timestamp_weekend', 'timestamp_dow')
    # Get the hour of each timestamp
    df = DataConversions.dt_hour(df, 'timestamp_hour', 'timestamp')

    return df


def clean_forecast(jwf):
    """Clean weather forecast data."""
    jwf.loc[:, 'date'] = pd.to_datetime(jwf.loc[:, 'date'], utc=True)

    # Convert timestamp to datetime for when the weather forecast was taken
    jwf = DataConversions.pd_datetime(jwf, 'timestamp')
    # Get the days_from by subtracting forecast day from timestamp
    jwf['days_from'] = (jwf['date'] - jwf['timestamp']).dt.days + 1
    # Convert date to datetime.date
    jwf = DataConversions.dt_date(jwf, 'date')

    return jwf


def merge_teetimes_forecast(df_t, jwf):
    """Function for merging, cleaning dataframes."""
    today = date.today()
    # Boolean mask for date range - after sept. 15th
    mask = df_t['date'] > date(2020, 9, 15)
    df_tf = df_t.loc[mask].copy(deep=True)

    # Drop duplicates from the teetime dataset
    df_tf.drop_duplicates(subset=["date", "time", "golfcourse_id"],
                          inplace=True)

    # Merge dataframes based on date, golfcourse_id
    df_t_jwf = pd.merge(df_tf, jwf, how="left", on=['date', 'golfcourse_id'],
                        indicator=True)

    # Only keep the rows that show up in both dataframes
    df_t_jwf = df_t_jwf.loc[df_t_jwf['_merge'] == "both"]
    df_t_jwf.drop(columns=['_merge', 'id_x', 'id_y', "times_updated",
                           'course_name', 'timestamp_x', 'timestamp_y',
                           'span_from', 'date_weekday'], inplace=True)

    df_t_jwf.drop_duplicates(subset=["date", "time", "golfcourse_id"],
                             inplace=True)

    save_the_date = df_t_jwf['date'].copy(deep=True)
    save_the_gcid = df_t_jwf['golfcourse_id'].copy(deep=True)

    df_output = df_t_jwf[['date', 'time', 'days_from_y', 'golfcourse_id']]
    mask_output = (df_output['date'] > today)
    df_output = df_output.loc[mask_output]

    return save_the_date, save_the_gcid, df_output, df_t_jwf


def clean_and_aggregate(df_train):
    """Split train/forecast, get aggregate counts, then merge."""
    ###########################################################################
    # Our dependent variable
    ###########################################################################
    var_dep = 'counts_daily'

    today = date.today()
    # Boolean mask dates from sept. 15 to today for training data split
    # Dates > today is used to make predictions
    mask_train = ((df_train['date'] > date(2020, 9, 15))
                  & (df_train['date'] <= today))
    df_train = df_train.loc[mask_train].copy(deep=True)

    # Get the aggregate number of teetimes for each day
    date_course_agg = df_train.groupby(
                                        ['date', 'golfcourse_id']
                                    ).size().reset_index()
    # Rename 3rd column to counts_daily
    date_course_agg.rename(columns={
                                    date_course_agg.columns[2]: "counts_daily"
                                    },
                           inplace=True)

    # Merge dataframes based on date, golfcourse_id
    df_train = pd.merge(df_train, date_course_agg,
                        on=['date', 'golfcourse_id'])

    # Drop the golfcourse_id and date columns
    df_train.drop(columns=['date', 'golfcourse_id'], inplace=True)

    vars_ind_train = df_train.columns.tolist()
    vars_ind_train.remove(var_dep)

    return df_train


class RemoteRequests():
    """Helper for accessing data from API."""

    @staticmethod
    def teetime_data(request_url):
        """Get request for teetime data from remote server."""
        payload = json.dumps({'code': '****'})

        response = requests.get(request_url, data=payload)
        # Convert response to json with defined column names
        df_t = pd.DataFrame(response.json(), columns=[
                            'id', 'golfcourse_id', 'course_name', 'time',
                            'date', 'timestamp', 'hour', 'week', 'month',
                            'year', 'times_updated'])

        # Call the clean_teetimes function
        df_t = clean_teetimes(df_t)
        df_t['time'] = df_t['time'].astype(str).str[:-3]
        return df_t

    @staticmethod
    def weather_forecast(request_url):
        """Get request - weather forecast from remote server."""
        # All courses
        payload = json.dumps({'code': '****'})

        wf = requests.get(request_url, data=payload)
        jwf = pd.DataFrame(wf.json())
        # Call clean_forecast function for data Transformations
        jwf = clean_forecast(jwf)

        return jwf


class DataConversions():
    """Data Cleaning Implementation."""

    @staticmethod
    def drop_duplicates(df, subset_list):
        """Subset should be a list of form ['var1', 'var2', 'var3']."""
        df.drop_duplicates(subset=subset_list, inplace=True)
        return df

    @staticmethod
    def pd_datetime(df, str_column):
        """Convert the column to datetime."""
        df.loc[:, str_column] = pd.to_datetime(df.loc[:, str_column])
        return df

    @staticmethod
    def dt_date(df, str_column):
        """Convert column to dt.date."""
        df[str_column] = df[str_column].dt.date
        return df

    @staticmethod
    def dt_time(df, str_column):
        """Convert column to dt.time."""
        df[str_column] = df[str_column].dt.time
        return df

    @staticmethod
    def if_weekend(df, to_column, from_column):
        """Creates a new onehot column (to_column) for weekend, from_column."""
        # df[to_column] = [(1 if (t in (5, 6) else 0)
        #                for t in df.loc[:, from_column]]
        return df

    @staticmethod
    def dt_dayofweek(df, to_column, from_column):
        """Convert column to dt.dayofweek."""
        df[to_column] = df[from_column].dt.dayofweek
        return df

    @staticmethod
    def dt_day_name(df, to_column, from_column):
        """Convert column to dt.day_name."""
        df[to_column] = df[from_column].dt.day_name
        return df

    @staticmethod
    def dt_days(df, to_column, from_column):
        """Convert column to dt.days."""
        df[to_column] = df[from_column].dt.days
        return df

    @staticmethod
    def dt_hour(df, to_column, from_column):
        """Convert column to dt.hour."""
        df[to_column] = df[from_column].dt.hour
        return df


class CategoricalVariables():
    """Implementation for Categorical Variables."""

    ###########################################################################
    # One-hot encoding of cateogorical variables
    ###########################################################################

    # Define the categorical columns to convert
    cols_to_convert = ['golfcourse_id', 'time', 'date', 'hour', 'week',
                       'month', 'year', 'date_dow',
                       'days_from_x', 'days_from_y', 'timestamp_dow',
                       # 'timestamp_weekend','date_weekend',
                       'timestamp_hour',
                       'condition', 'zip_code']
    vars_ind_onehot = []

    df_t_jwf, vars_ind = None, None

    def __init__(self, df_t_jwf):
        df_t_jwf[self.cols_to_convert] = df_t_jwf[self.cols_to_convert]\
                                                      .astype('object')
        vars_ind_categorical = (df_t_jwf.select_dtypes(
                                    include=['object']
                                    ).columns.tolist())
        # Get all the column names from dataframe with teetimes
        # And weather forecast data
        # vars_all = df_t_jwf.columns.values

        # df_all_onehot = df_t_jwf.copy(deep=True)
        # Return one-hot encoded dataframe
        self.df_t_jwf = self.convert_to_onehot(
                                                self,
                                                df_t_jwf,
                                                vars_ind_categorical
                                                )
        # Get a list of the independent variables
        self.vars_ind = self.df_t_jwf.columns.tolist()

    @classmethod
    def convert_to_onehot(cls, self, df_t_jwf, vars_ind_categorical):
        """Return one hot encoded dataframe."""
        for col in vars_ind_categorical:
            # use pd.get_dummies on  df_all[col]
            df_oh = pd.get_dummies(df_t_jwf[col], drop_first=False)
            # Rename the columns to have the original variable name as a prefix
            oh_names = [col + '_' + str(c) for c in df_oh.columns]
            df_oh.columns = oh_names

            df_t_jwf = pd.concat(
                                [df_t_jwf, df_oh],
                                axis=1, sort=False
                                )
            del df_t_jwf[col]
            self.vars_ind_onehot.extend(oh_names)

        # rename df_all_onehot to df_all as this is now the data we will be
        # using for the rest of this work
        # df_t_jwf = df_all_onehot.copy(deep=True)

        # del df_all_onehot
        return df_t_jwf

    def categorical_dataframe(self):
        """Return dataframe."""
        return self.df_t_jwf

    def independent_vars(self):
        """Return list of independent variables."""
        return self.vars_ind
