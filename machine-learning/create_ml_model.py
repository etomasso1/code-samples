import os
from datetime import date

import numpy as np

from catboost import CatBoostRegressor


class MLModel():
    """Train and Validate ML Implementation."""

    # Our dependent variable
    var_dep = 'counts_daily'

    today = date.today()

    # Output
    df_preds = None

    def __init__(self, df_train, df_predict,
                 df_output, vars_ind, save_the_date):
        # Where df is df_train, df_predict is the pre-onehot dataframe
        # df_output is a previously created dataframe for prediction values
        # save_the_date is a column of dates from pre-transformation

        # Split into train, test folds
        df_train, dict_of_folds = self.train_split(self, df_train, vars_ind)
        # Create the model with static params
        model_regress = self.define_model()
        # Fit the model
        model_regress.fit(dict_of_folds['X_design'],
                          dict_of_folds['y_design'],
                          eval_set=(
                                    dict_of_folds['X_test'],
                                    dict_of_folds['y_test']
                                    ),
                          logging_level='Silent',
                          plot=False)

        # Save model to local directory
        self.save_model(model_regress)
        # Class: Get the dataframe of X variables used to make predictions
        X_predict = self.df_predict_staging(self, df_predict, save_the_date)
        # Make predictions from regression model
        predictions = model_regress.predict(X_predict)

        # Format predictions
        self.df_preds = self.df_preds_formatting(df_output, predictions)
        # Save predictions to local directory
        self.save_predictions_locally(self.df_preds)

    @classmethod
    def train_split(cls, self, df_train, vars_ind):
        """Set train, test split folds."""
        rng = np.random.RandomState(2020)
        df_train['fold'] = rng.randint(0, 10, df_train.shape[0])

        # define index for train, val, design, test
        idx_train  = np.where(df_train['fold'].isin(np.arange(0, 6)))[0]
        idx_val    = np.where(df_train['fold'].isin([6, 7]))[0]
        idx_design = np.where(df_train['fold'].isin(np.arange(0, 8)))[0]
        idx_test   = np.where(df_train['fold'].isin([8, 9]))[0]

        X = df_train[vars_ind].values
        y = df_train[self.var_dep].values

        X_train  = X[idx_train, :]
        X_val    = X[idx_val, :]
        X_design = X[idx_design, :]
        X_test   = X[idx_test, :]

        y_train  = (df_train[self.var_dep].iloc[idx_train]
                                          .copy().values.ravel())
        y_val    = df_train[self.var_dep].iloc[idx_val].copy().values.ravel()
        y_design = (df_train[self.var_dep].iloc[idx_design]
                                          .copy().values.ravel())
        y_test   = df_train[self.var_dep].iloc[idx_test].copy().values.ravel()

        X_train = X_train.astype(int)
        y_train = y_train.astype(int)
        X_val = X_val.astype(int)
        y_val = y_val.astype(int)
        X_design = X_design.astype(int)
        y_design = y_design.astype(int)
        X_test = X_test.astype(int)
        y_test = y_test.astype(int)

        dict_of_folds = {
                        'X_design': X_design,
                        'y_design': y_design,
                        'X_test': X_test,
                        'y_test': y_test
                        }

        return df_train, dict_of_folds

    @classmethod
    def define_model(cls):
        """Model params."""
        model_regress = CatBoostRegressor(
              iterations=4000,
              use_best_model=True,
              eval_metric='R2',
              od_type='Iter',
              od_wait=1000,
              boosting_type='Plain',
              bootstrap_type='Bernoulli',
              one_hot_max_size=10,
              depth=6,
              l2_leaf_reg=3,  # default is 3
              loss_function='RMSE',
              random_seed=2020,
              logging_level='Silent'
              )
        return model_regress

    @classmethod
    def save_model(cls, model_regress):
        """Save the model to local directory."""
        os_models = "****"
        model_regress.save_model(os.path.join(os_models, 'cb1_wf'))

    @classmethod
    def df_predict_staging(cls, self, df_predict, save_the_date):
        """Cleaning, data processing for predictions dataframe."""
        # Date column is the pre-transformation date column
        df_predict['date'] = save_the_date

        # Boolean mask for date range
        df_mask = (df_predict['date'] > self.today)
        # Apply boolean mask
        df_predict = df_predict.loc[df_mask]
        # Drop the date column that was previously added
        df_predict.drop(columns=['date'], inplace=True)
        # Convert col
        vars_predict = df_predict.columns.tolist()

        # Get the values of the X_predictions dataframe
        X_predict = df_predict[vars_predict].values

        return X_predict

    @classmethod
    def df_preds_formatting(cls, df_output, predictions):
        """Formatting the predictions for golfcourse use."""
        df_output['preds'] = predictions.round(0)
        df_preds = (df_output.groupby(['date', 'golfcourse_id'])['preds']
                             .mean().round(0).reset_index())
        return df_preds

    @classmethod
    def save_predictions_locally(cls, df_preds):
        """Save the predictions to local directory."""
        os_results = "****"
        fname = 'df_preds-' + str(date.today()) + '.pkl'

        df_preds.to_pickle(os.path.join(os_results, fname))

    def return_df_preds(self):
        """Return dataframe of predictions."""
        return self.df_preds
