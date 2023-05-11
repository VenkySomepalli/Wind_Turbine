# Wind_Turbine
Problem Statement:
The intermittent nature and low control over the wind conditions bring up the same problem to every grid operator in their successful integration to satisfy current demand. In combination with having to predict demand and balance it with the supply, the grid operator now also must predict the availability of wind and solar generation plants in the next hour, day, or week. Apart from holding back the benefits of renewable energy, incorrectly scheduling of wind generation plants may lead to unnecessary reservations, higher costs passed over to the consumer, and use of other more expensive and polluting power resources.
Working with real data is challenging due to noise and missing periods.

# Dataset details:
The provided full-year hourly time-series are simulated using the National Renewable Energy Laboratory (NREL) software for a location in Texas, US. It has perfect data completeness, and no noisy data; challenges that hinder forecasting tasks with real datasets and distract from the goal. The dataset contains various weather features which can be analyzed and used as predictors.

# Column names:
Time stamp

System power generated | (kW)

Wind speed | (m/s)

Wind direction | (deg)

Pressure | (atm)

Air temperature | ('C)

# Aim:
Predict System Power Generated per kW by using above features.

# Performed EDA(Exploratory Data Analysis) manually:
Manuvally checked graphically representation; univariate, bivariate, multivariate plots, and also 
checked 1st, 2nd, 3rd and 4th moment Business Decision for all features.
# Performed AutoEDA:
Checked Sweetviz, PandasProfiling, Autoviz, D-tale, DataPrep

# Data Cleansing/Data Preparation/Data Munging/Data Wangling/ Data Organizing:
Done all the necessary data cleaning process(refer to Python file)

# Model Building:
Manually used all the regression models including tree based models.

Applied AutoMl: TPOT Regressor
Both have performed weel got test ans train accuracies are nearly 99%.
Accuracies of different models:
![image](https://github.com/VenkySomepalli/Wind_Turbine/assets/106543953/1ad8b429-9bd3-4cdd-b72c-6fc94c4480a4)


See the below image predicted and actual values.

![image](https://github.com/VenkySomepalli/Wind_Turbine/assets/106543953/72458688-574b-4fb4-9614-2cf93107a654)


