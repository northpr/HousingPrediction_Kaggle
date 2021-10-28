# Data Science Housing Prediction Project
- Create a tool that estimates the house cost decided by many variables such as area, number of floors, location, and many other variables by using feature engineering
- Create more variables from the previous to have more meaningful data to analyze further.
- Create a basic model for use and test to use in further

## Code and Resources Used
**Python Version**: 3
**Packages**: numpy, pandas, seaborn, matplotlib, sklearn, xgboost and more

## Basic Virtualization
To understand the model we want to predict better, I have done fundamental analysis to understand better, clean and otherwise. I will give some example

To remove outliers and to understand SalePrice.
![SalePrice](https://github.com/northpr/HousingPrediction_Kaggle/blob/main/images/Screen%20Shot%202564-10-28%20at%2000.00.13.png?raw=true)
```Python
sns.scatterplot(x='GrLivArea',y='SalePrice',data=df_train)
```

To have a better understanding of the correlation of factors.
![Correlation](https://github.com/northpr/HousingPrediction_Kaggle/blob/main/images/Screen%20Shot%202564-10-28%20at%2000.00.32.png?raw=true)
```Python
plt.subplots(figsize=(12,9))
sns.heatmap(data=df_train.corr(),square=True)
```


## Data Cleaning Task
After digging deep into the data and analysing it, I needed to clean it to use it in our model. I made many changes such as:
- Clean missing data by using Null, Mean, and other factors depending on statistics.
- Create a column by feature engineering such as Area, GarageArea, GrLivArea, Remodeled and more.
- Remove some columns that might affect the model result.
- Remove outliers that can have a negative effect on the model we will build.

Example of fillina some missing data in NA
```Python
data['PoolQC']=data['PoolQC'].fillna('NA')
data['MiscFeature']=data['MiscFeature'].fillna('NA')
data['Alley']=data['Alley'].fillna('NA')
data['Fence']=data['Fence'].fillna('NA')
data['FireplaceQu']=data['FireplaceQu'].fillna('NA')
```

Example of feature engineering to create a new variable called 'Remodeled'
```Python
data['Remodeled']=0
data.loc[data['YearBuilt']!=data['YearRemodAdd'],'Remodeled']=1
data=data.drop('YearRemodAdd',1)
data=pd.get_dummies(data=data,columns=['Remodeled'])
```

## Model Building
First, I transformed the categorical variables into dummy variables. Then, I also split the data into train and test with a test size of 30%.

```Python
data['BsmtExposure'] = data['BsmtExposure'].map({'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'NA':0})
data['CentralAir'] = data['CentralAir'].map({'Y':1, 'N':0})
data['GarageFinish'] = data['GarageFinish'].map({'Fin':3, 'RFn':2, 'Unf':1, 'NA':0})
data['PavedDrive'] = data['PavedDrive'].map({'Y':1, 'P':0.5, 'N':0})
data['BsmtFinType1'] = data['BsmtFinType1'].map({'GLQ':5, 'ALQ':4, 'BLQ':3, 'Rec':2, 'LwQ':1, 'Unf':0})
data['BsmtFinType2'] = data['BsmtFinType2'].map({'GLQ':5, 'ALQ':4, 'BLQ':3, 'Rec':2, 'LwQ':1, 'Unf':0})
data['Electrical'] = data['Electrical'].map({'SBrkr':4, 'FuseA':3, 'FuseF':2, 'FuseP':1, 'Mix':0})
data['Fence'] = data['Fence'].map({'GdPrv':4, 'MnPrv':3, 'FdWo':2, 'MnWw':1, 'NA':0})
```

## Model Performance
I created a model and used R-squared and RMSE to see how effectively the model is, which gives R-squared at 0.95, almost close to 1 for linear regression, but Random Forest is the best R-squared at 0.98. The last one is XGBoost, and the R-squared is at 0.99, which is only 0.01, close to 1.

XGBoost Model which R-square is = 0.9912
```Python
from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(x_train_scaled, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(x_train_scaled, y_train)], 
             verbose=False)
print(my_model.score(x_train_scaled,y_train))
```
Furthermore please check my .ipynb and thanks for viewing
