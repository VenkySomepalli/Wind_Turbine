# -*- coding: utf-8 -*-
"""
Created on Sun May  7 16:29:19 2023

@author: venki
"""
################ Texas wind power Analysis and Model building #########

import numpy as np ## Mathematical Or numerical calcumation
import pandas as pd ## Data modifications

## Load the data set
wind = pd.read_csv('C:/Users/venki/OneDrive/Desktop/Datascience360/Texas_wind_turbine/TexasTurbine.csv')

wind.head()

wind.shape  ## shape (rows, columns) = (8760, 6)
wind.info()

## time stamp in the form of object convert to datetime

wind['Time stamp'] = pd.to_datetime(wind['Time stamp'], format = '%b %d, %I:%M %p')

wind.info()

wind.describe() ## statistical information

## column names have spaces, special characterstics for feature purpose rename the columns;
wind.columns
''' ['Time stamp', 'System power generated | (kW)', 'Wind speed | (m/s)',
       'Wind direction | (deg)', 'Pressure | (atm)', 'Air temperature | ('C)']'''

wind.rename(columns = {'Time stamp' : 'Time_stamp', 'System power generated | (kW)' : 'SPG_kW',
                       'Wind speed | (m/s)' : 'WS_ms', 'Wind direction | (deg)' : 'WD_deg',
                       'Pressure | (atm)' : 'Pressure_atm', 'Air temperature | (C)' : 'AT_C' }, inplace = True)

wind["Month"] = wind["Time_stamp"].dt.month

################## EDA(Exploratory Data Analysis) ################
## Univariate Graphs

import matplotlib.pyplot as plt
import seaborn as sns

#Strip plot
sns.stripplot(wind['SPG_kW'])
sns.stripplot(wind['WS_ms'])
sns.stripplot(wind['WD_deg'])
sns.stripplot(wind['Pressure_atm'])
sns.stripplot(wind['AT_C'])
''' Not getting any information, all the data lie on some range of values'''

# Swarm plot
sns.swarmplot(wind['SPG_kW'])
sns.swarmplot(wind['WS_ms'])
sns.swarmplot(wind['WD_deg'])
sns.swarmplot(wind['Pressure_atm'])
sns.swarmplot(wind['AT_C'])

''' Same as the stripplot'''

# Histogram (plt.hist or sns.distplot)

plt.hist(wind['SPG_kW']) # right skewed
plt.hist(wind['WS_ms']) # symmetrically distributed
plt.hist(wind['WD_deg']) # nearly symmetric distributed
plt.hist(wind['Pressure_atm']) # Symmetrically distributed
plt.hist(wind['AT_C']) # slighlty symmetrically distributed

sns.distplot(wind['SPG_kW'], kde = True, color = 'blue', bins = 10)
sns.distplot(wind['WS_ms'], kde = True, color = 'blue', bins = 10)
sns.distplot(wind['WD_deg'], kde = True, color = 'blue', bins = 10)
sns.distplot(wind['Pressure_atm'], kde = True, color = 'blue', bins = 10)
sns.distplot(wind['AT_C'], kde = True, color = 'blue', bins = 10)

#Density plots
wind['SPG_kW'].plot( kind = 'density')
wind['WS_ms'].plot( kind = 'density')
wind['WD_deg'].plot(kind = 'density')
wind['Pressure_atm'].plot( kind ='density')
wind['AT_C'].plot(kind = 'density')

sns.kdeplot(wind['SPG_kW'], shade = True)
sns.kdeplot(wind['WS_ms'], shade = True)
sns.kdeplt(wind['WD_deg'], shade = True)
sns.kdeplt(wind['Pressure_atm'], shade = True)
sns.kdeplot(wind['AT_C'], shade = True)

## Box plot
plt.boxplot(wind['SPG_kW'])
plt.boxplot(wind['WS_ms'])
plt.boxplot(['WD_deg'])
plt.boxplot(['Pressure_atm'])
plt.boxplot(['AT_C'])

sns.boxplot(wind['SPG_kW'], orient = 'h', color = 'Violet')
sns.boxplot(wind['WS_ms'], orient = 'h', color = 'Indigo')
sns.boxplot(wind['WD_deg'], orient = 'h', color = 'Blue')
sns.boxplot(wind['Pressure_atm'], orient = 'h', color = 'Green')
sns.boxplot(wind['AT_C'], orient = 'h', color = 'Red')

sns.boxplot(data = wind, orient ="h", palette = "Set2") # all in one go

## Violin plots
plt.violinplot(wind.SPG_kW, showmedians = True)
plt.violinplot(wind.WS_ms, showmedians = True)
plt.violinplot(wind.WD_deg, showmedians = True)
plt.violinplt(wind.Pressure_atm, showmedians = True)
plt.violinplot(wind.AT_C, showmedians = True)

sns.violinplot(wind['SPG_kW'], orient = 'vertical')
sns.violinplot(wind['WS_ms'], orient = 'vertical')
sns.violinplot(wind['WD_deg'], orient = 'vertical')
sns.violinplot(wind['Pressure_atm'], orinet = 'vertical')
sns.violinplot(wind['AT_C'], orient = 'vertical')

sns.violinplot(x = wind['SPG_kW'], y = wind['WS_ms'], data = wind)

## Categorical Variables visulization

# Bar chart
wind['column name'].value_counts().plot.bar()

sns.counterplot(wind['column name'])

# Pie chart
plt.pie(wind['column name'].value_counts(), labels=['','',''])


#### Bivariate plots

## Sctter plots
wind[wind['WS_ms'] < 100].sample(100).plot.scatter(x = 'WS_ms', y = 'Pressure_atm')

wind[wind['WS_ms'] < 100].plot.scatter(x = 'WS_ms', y = 'Pressure_atm')

## Hexaplot
wind[wind['WS_ms'] < 100].sample(100).plot.hexbin(x = 'WS_ms', y = 'Pressure_atm', gridsize = 15)

## Barplot; it works only numerics
wind.drop(['Time_stamp'], axis = 1, inplace = True)
wind.plot.bar(stacked = True)


wind.plot.area()

wind.plot.line()

## Multivariate  plots.
sns.pairplot(wind, hue = 'WS_ms')

sns.heatmap(wind, annot = True, fmt = ".2f")
sns.heatmap(wind, annot = wind.rank(axis = "columns"))


## For different type of plots follow the seaborn link:https://seaborn.pydata.org/generated/seaborn.barplot.html

################ Auto EDA ####################
## Follow different type auto EDA
# 1. Sweetviz
''' install: pip install sweetviz'''
import sweetviz as sv
s = sv.analyze(wind)
s.show_html()

# 2. Autoviz
''' install: pip install autoviz'''
from autoviz.AutoViz_Class import AutoViz_Class
av = AutoViz_Class()
a = av.AutoViz(" path to or data")

# 3. D-Tale
''' install: pip install dtale '''
import dtale
d = dtale.show(wind)
d.open_browser()

# 4. Pandas Profiling
''' install : pip install pandas_profiling'''
from pandas_profiling import ProfileReport
p = ProfileReport(wind)
p.to_file("output.html")

# 5. DataPrep
''' install: pip install dataprep'''
from dataprep.eda import create_report
report = create_report(wind, title = 'My Report')
report.show_browser()


############## Data cleaning ##########


# 1. Typecasting (converting one data type to another)
wind.dtypes # know the type of data

# convert float to integer
wind.astype('int64').dtypes
# convert integer to float
wind.astype('float64').dtypes

# we can convert Array to series or Dataframe.

# 2. Handling duplicates.

wind.duplicated() # False means no duplicates
wind.duplicated().sum() # sum of the duplicates

# Remove the duplicates
wind.drop_duplicates() # delete the duplicates
wind.columns

# 3. Out lier treatment
# find the outliers in features using box plot.
cols = ['SPG_kW', 'WS_ms', 'WD_deg', 'Pressure_atm', 'AT_C']
# for loop for the above cols to plot the boxplot.
for i in cols:
    sns.boxplot(wind[i]); plt.show() # We have a outliers

## Treatment of the outliers.

# Detection of outliers (find limits for salary based on IQR)
percentile75 = wind['WS_ms'].quantile(0.75)
percentile25 = wind['WS_ms'].quantile(0.25)

IQR = wind['WS_ms'].quantile(0.75) - wind['WS_ms'].quantile(0.25)
lower_limit = wind['WS_ms'].quantile(0.25) - (IQR * 1.5)  ## defaut value is 3.0 in place of 1.5
upper_limit = wind['WS_ms'].quantile(0.75) + (IQR * 1.5)

## Three different methods to treat outliers; Remove, Replace and Retain
################# 1.Remove (trim the dataset) ##########
# Trimming Technique
#let's flag the outliers in the data set

outliers_df = np.where(wind['WS_ms'] > upper_limit, True, np.where(wind['WS_ms'] < lower_limit, True, False))
wind_trimmed =  wind.loc[~(outliers_df),]
wind.shape, wind_trimmed.shape

### lets explore outliers in the trimmed dataset
sns.boxplot(wind_trimmed.WS_ms)  ## we got no out liers

############## 2. Replace #########
# Now let's replace the outliers by the maximum and minimum limit
wind['wind_replaced'] = pd.DataFrame(np.where(wind['WS_ms'] > upper_limit, upper_limit, np.where(wind['WS_ms'] < lower_limit, lower_limit, wind['WS_ms'])))
sns.boxplot(wind.wind_replaced)

########### 3. Winsorization (Retain) ############
from feature_engine.outliers import Winsorizer

winsor = Winsorizer(capping_method = 'iqr' , # choose IQR rule boundaries or gaussian for mean std
                    tail = 'both', # cap left, right or both tails
                    fold = 1.5,
                    variables = ['WS_ms'])
wind_t = winsor.fit_transform(wind[['WS_ms']])

# We can inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

#lets see boxplot
sns.boxplot(wind_t.WS_ms)


## I felt winsorization is good technique to treat outliers for this data.

#Outlier treatment with winsorization
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method ='iqr', # choose IQR rule boundaries or gaussian for mean and std
                   tail = 'both', # cap left, right or both tails
                   fold = 1.5,
                  # variables = ['']
                  )

for i in cols:
    wind[i] = winsor.fit_transform(wind[[i]])

print("left", winsor.left_tail_caps_)
print("right", winsor.right_tail_caps_)

bx_wind = sns.boxplot(data = wind, orient = "h", palette = "Set2" )

## 4. Variance
''' If the variance is low or close to zero, then the feature is approximately constanst
and will not improve the model accuracy, in that case drop the feature.'''
wind.var()

## 5. Missing vlues

# Find the missing values
wind.isna().sum() # No missng values.


# If missing values exists, follow the procedure to remove or replace.

# Delete missing values
wind.dropna().sum()

########## Simple Imputation Methods
## Create an imputer object that fills 'Nan' values
# Mean and Median imputer are used for numeric data (Salaries)
# Mode is used for discrete data (ex: Position, Sex, MaritalDesc)

# for Mean, Meadian, Mode imputation we can use Simple Imputer or df.fillna()
from sklearn.impute import SimpleImputer

#### Mean Imputer (Generally, data have no outliers)
mean_imputer = SimpleImputer(missing_values = np.nan , strategy='mean')
wind["WS_ms"] = pd.DataFrame(mean_imputer.fit_transform(wind[["SWS_ms"]]))
wind["WS_ms"].isna().sum()

# Median Imputer (outliers exists)
median_imputer = SimpleImputer(missing_values=np.nan, strategy ='median')
wind["WS_ms"] = pd.DataFrame(median_imputer.fit_transform(wind[["WS_ms"]]))
wind["WS_ms"].isna().sum() ## all records replaced by median

wind.isna().sum()

## Mode Imputer (categorical data)
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
wind["WS_ms"] = pd.DataFrame(mode_imputer.fit_transform(wind[["WS_ms"]]))
wind.isna().sum() # all categorical records replaced by mode

### Random Imputation
from feature_engine.imputation import RandomSampleImputer
imputer = RandomSampleImputer(random_state = ['cat1/numeric1', 'cat2/numeric2'],
                              seed = 'observation', seeding_method = 'add')
wind['WS_ms'] = pd.DataFrame(imputer.fit_transform(wind[['WS_ms']]))


### KNN Imputation
from skelarn.impute import KNNImputer
imputer = KNNImputer(n_neighbors = 2, weights = "uniform")
wind['WS_ms'] = imputer.fit_transform(wind[['WS_ms']])

### 6. Discretization / Binning / Grouping.
''' Discretization: Cnverting continuous data to discrete data
  Binarization: Converting continuous data into 2 categories
  Rounding: Rounding to nearest value
  Binning: Fixed width Binning/ Adaptive Binning '''
  
wind['WS_ms'] = pd.cut(wind['WS_ms'], bins=[min(wind.WS_ms) - 1, wind.Ws_ms.mean(), max(wind.WS_ms)], labels=["Low", "High"])
  #or
wind['binned']  = pd.cut(x = wind['WS_ms'], bins = [0, 25, 50,100])
#or
wind['binned']  = pd.cut(x = wind['WS_ms'], bins = [0, 25, 50,100], labels = [0, 1, 2])

### 7. Dummy Variable Creation

## One-Hot Encoding

from sklearn.preprocessing import OneHotEncoder
# Creating instance of One Hot Encoder
enc = OneHotEncoder() # initializing method

enc_df = pd.DataFrame(enc.fit_transform(wind.iloc[:, 2:]).toarray())

## Label Encoding
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()
wind['WS_ms'] = label_encoder.fit_transform(wind['WS_ms'])

## Dummy Coding Scheme
pd.get_dummies(wind)

## Column transformer # Combined few codings(encoders, normalization..)
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MinMaxScaler
ct = ColumnTransformer([("text_preprocess", FeatureHasher(input_type = "string"), "documents"),
                       ("num_preprocess", MinMaxScaler(), ["width"])])
X_trans = ct.fit_transform(wind)


### 8. Transformation
# Chek normala distribution using Q-Q plot

# Apply tranformations like log, sqrt, cbrt, inversesqrt, BoxCox ...etc.

### 9. Feature Scaling/Feature Shrinking

#Standardized Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaled = scaler.fit_transform(wind)
print(scaled)

# Normalization/Min-MaxScaling.

from sklearn.preprocessing import MinMaxScaler
trans = MinMaxScaler()
scaled = trans.fit_transform(wind)
print(scaled)

# Robust Scaling (follows IQR)
from sklearn.preprocessing import RobustScaler
robust =  RobustScaler()
scaled = robust.fit_transform(wind)
print(scaled)

### 10. String Manipulations
''' Text modification and cleaning need to apply;Tokenization,
Stemming/Lemmatization, Stopword Removal, Document similarity, Topic Models, Word Embeddings '''


############### Model Building ##############

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, confusion_matrix, accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, minmax_scale, MaxAbsScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

wind.set_index("Time_stamp", inplace = True)
wind.head()

# Defined X value and y value, and split the data train
X = wind.drop(columns = "SPG_kW")
y = wind["SPG_kW"] # System power generated

#Split the data train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42 )

# Defined object from library regression
LR  = LinearRegression()
DTR = DecisionTreeRegressor()
RFR = RandomForestRegressor()
KNR = KNeighborsRegressor()
MLP = MLPRegressor()
XGB = XGBRegressor()
SVR = SVR()

# Make for loop for regression

linear = [LR, DTR, RFR, KNR, MLP, XGB, SVR]
d = {}
for i in linear:
    i.fit(X_train, y_train)
    y_pred = i.predict(X_test)
    print(i, ":", r2_score(y_test, y_pred)*100)
    d.update({str(i):i.score(X_test, y_test)*100})
    
# plot for accuracy

plt.title("Algorithm vs Accuracy")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.plot(d.keys(), d.values(), marker = 'o', color = 'blue')
plt.show()
''' Best model is XGBRegressor '''

############ AutoML ##############

## Applied TOPT Regressor and got the best best model.
from sklearn.metrics import mean_squared_error, make_scorer
rmse = lambda y, y_hat: np.sqrt(mean_squared_error(y, y_hat))

from tpot import TPOTRegressor

rmse_scorer = make_scorer(rmse, greater_is_better = False)
pipeline_optimizer = TPOTRegressor(
    scoring = rmse_scorer,
    max_time_mins = 60,
    random_state = 42,
    verbosity = 2
    )
pipeline_optimizer.fit(X_train, y_train)

print(pipeline_optimizer.score(X_test, y_test))

pipeline_optimizer.fitted_pipeline_
pipeline_optimizer.export('wind_turbine.py')

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from tpot.export_utils import set_param_recursive

# Average CV score on the training set was: -5.664466737080142
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    ExtraTreesRegressor(bootstrap=False, max_features=0.45, min_samples_leaf=1, min_samples_split=13, n_estimators=100)
)

exported_pipeline.fit(X_train, y_train)

y_pred_test = exported_pipeline.predict(X_test)

result_test = pd.DataFrame({'Actual':y_test, "Predicted": y_pred_test})
result_test.head(10)

## importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# predicting the accuracy score
score_test = r2_score(y_test, y_pred_test)

print('R2 score(test): ', score_test)
print('Mean squared error(test): ', mean_squared_error(y_test, y_pred_test))
print('Root Mean squared error(test): ', np.sqrt(mean_squared_error(y_test, y_pred_test)))

""" 
R2 score(test):  0.9999688122448446
Mean squared error(test):  23.278107957390734
Root Mean squared error(test):  4.824739159518443
"""

y_pred_train = exported_pipeline.predict(X_train)

result_train = pd.DataFrame({'Actual':y_train, "Predicted": y_pred_train})
result_train.head(10)

score_train = r2_score(y_train, y_pred_train)

print('R2 score(train): ', score_train)
print('Mean squared error(train): ', mean_squared_error(y_train, y_pred_train))
print('Root Mean squared error(train): ', np.sqrt(mean_squared_error(y_train, y_pred_train)))

"""
R2 score(train):  0.9999831993825371
Mean squared error(train):  13.148067413756223
Root Mean squared error(train):  3.626026394520071
"""
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred_test)










