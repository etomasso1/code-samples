All private information has been replaced by ##########

The file demand-prediction.py follows the following steps:

1. CeleryBeat runs the script once a day and checks if a .pkl file exists with today's datetime.date in the filename. If not:
2. GET requests are served to a demand data and weather forecast API endpoint set up on the application and parsed as json data.
3. Demand data is cleaned, timestamps and str date and time data is converted to datetime.date and datetime.time
4. Demand and weather forecast data are merged based on date, golfcourse_id.
5. All data from the beginning of October to today's current date is used as the train validation. Daily counts are created for each date.
6. Categorical variables are one-hot encoded. Train and validation sets are set up with the training data. The transformations are also applied to the test set.
7. The Catboost regression model is defined, then fit. Predictions are created for each row of data and the mean daily count is calculated.
8. This data frame is then stored to a pickle file in a specified path.
