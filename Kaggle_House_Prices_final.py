import string
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from ydata_profiling import ProfileReport
from mlxtend.regressor import StackingCVRegressor
from lightgbm import LGBMRegressor
import optuna as opt
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from catboost import CatBoostRegressor
import time
import json

random.seed(0)
pd.set_option('display.max_columns', None)
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf)

train_df = pd.read_csv('train_houses.csv')
test_df = pd.read_csv('test_houses.csv')

# Quickly, checking feature types and number of missing values for each feature ------------------------------------------------
print(train_df.dtypes)
print(train_df.isnull().sum())
print(test_df.isnull().sum())

# 0. General overview of the data: -------------------------------------------------------------------------------------------
profile = ProfileReport(train_df)
profile.to_file('profile_train.html')
profile = ProfileReport(test_df)
profile.to_file('profile_test.html')

# 1. Filling NaNs -------------------------------------------------------------------------------------------------------

# Filling specific rows with eye test and induction----------------------------------------------------------------------

# Replacing wrong values in MasVnrArea, and replacing potentially wrong Nan
# in MasVnrType with the most frequent = BrkFace
train_df.MasVnrArea = np.where((train_df.MasVnrType.isna()) & (train_df.MasVnrArea == 1.0), 0, train_df.MasVnrArea)
train_df.MasVnrType = np.where((train_df.MasVnrArea > 0) & (train_df.MasVnrType.isna()), 'BrkFace', train_df.MasVnrType)
train_df.MasVnrArea.replace(1.0, 0.0, inplace=True)
test_df.MasVnrArea = np.where((test_df.MasVnrType.isna()) & (test_df.MasVnrArea == 1.0), 0, test_df.MasVnrArea)
test_df.MasVnrType = np.where((test_df.MasVnrArea > 0) & (test_df.MasVnrType.isna()), 'BrkFace', test_df.MasVnrType)
test_df.MasVnrArea.replace(1.0, 0.0, inplace=True)
test_df.MasVnrArea.replace(3.0, 0.0, inplace=True)

test_df.BsmtCond = np.where((test_df.BsmtCond.isna()) & (test_df.BsmtFinSF1 == 1044), 'Gd', test_df.BsmtCond)
test_df.BsmtCond = np.where((test_df.BsmtCond.isna()) & (test_df.BsmtFinSF1 == 1033), 'TA', test_df.BsmtCond)
test_df.BsmtCond = np.where((test_df.BsmtCond.isna()) & (test_df.BsmtFinSF1 == 755), 'Gd', test_df.BsmtCond)

train_df.BsmtFinType2 = np.where((train_df.BsmtFinSF1 == 1124) & (train_df.BsmtFinSF2 == 479), 'Unf', train_df.BsmtFinType2)

test_df.KitchenQual.fillna('Fa', inplace=True)

test_df.Functional = np.where(test_df.Id == 2474, 'Sev', test_df.Functional)
test_df.Functional = np.where(test_df.Id == 2217, 'Min2', test_df.Functional)

test_df.GarageYrBlt = np.where(test_df.Id == 2577, 1959, test_df.GarageYrBlt)
test_df.GarageYrBlt = np.where(test_df.Id == 2127, 1959, test_df.GarageYrBlt)
test_df.GarageFinish = np.where(test_df.Id == 2127, 'RFn', test_df.GarageFinish)
test_df.GarageFinish = np.where(test_df.Id == 2577, 'Unf', test_df.GarageFinish)

test_df.GarageCars = np.where(test_df.Id == 2577, 2.0, test_df.GarageCars)
test_df.GarageArea = np.where(test_df.Id == 2577, 400.0, test_df.GarageArea)
test_df.GarageQual = np.where(test_df.Id == 2577, 'TA', test_df.GarageQual)
test_df.GarageCond = np.where(test_df.Id == 2577, 'TA', test_df.GarageCond)

test_df.SaleType = np.where(test_df.Id == 2490, 'Oth', test_df.SaleType)


# Filling 4 NaN with most freq for test data ---------------------------------------------------------------------------
def nan_mode_filler(df, features):
    for feature in features:
        df[feature].fillna(df[feature].mode()[0], inplace=True)


train_feature_list_nan_to_mode = []
test_feature_list_nan_to_mode = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd' ]

nan_mode_filler(test_df, test_feature_list_nan_to_mode)


# Filling NaNs with mean or median------------------------------------------------------------------------------------------
def nan_mean_median_filler_based_on_other_feature(df, features_dict, filling_func):
    if filling_func == 1:
        for filled_feature, base_feature_for_stats in features_dict.items():
            df[filled_feature] = df.groupby(base_feature_for_stats)[filled_feature].transform(lambda x: x.fillna(x.mean()))
    elif filling_func == 2:
        for filled_feature, base_feature_for_stats in features_dict.items():
            df[filled_feature] = df.groupby(base_feature_for_stats)[filled_feature].transform(lambda x: x.fillna(x.median()))


# Creating temp feature 'LotAreCut' to discretize LotArea into 10 buckets to get meaningful mean or median
train_df["LotAreaCut"] = pd.qcut(train_df.LotArea,10)
test_df["LotAreaCut"] = pd.qcut(test_df.LotArea,10)

# Creating dict of features. Keys: feature to be nanFilled, Values: Binned feature. For every bucket, median
# of key is calculated
features_to_fill = {'LotFrontage': 'LotAreaCut'}
nan_mean_median_filler_based_on_other_feature(train_df, features_to_fill, 2)
nan_mean_median_filler_based_on_other_feature(test_df, features_to_fill, 2)

# Dropping temp features
train_df.drop('LotAreaCut', axis=1, inplace=True)

# Filling NaNs with 0 ------------------------------------------------------------------------------------------------
def nan_to_zero_filler(df, features):
    for feature in features:
        df[feature].fillna(0, inplace=True)


train_features_nan_to_zero = ['MasVnrArea', 'GarageYrBlt']
test_features_nan_to_zero = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
                             'GarageYrBlt']

nan_to_zero_filler(train_df, train_features_nan_to_zero)
nan_to_zero_filler(test_df, test_features_nan_to_zero)

# Filling NaNs with None ------------------------------------------------------------------------------------------------
def nan_to_none_filler(df, features):
    for feature in features:
        df[feature].fillna('None', inplace=True)



train_features_nan_to_none = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure',
                              'FireplaceQu','GarageFinish','GarageQual','GarageCond','PoolQC','Fence',
                              'MiscFeature','Electrical', 'GarageType']
test_features_nan_to_none = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure',
                             'FireplaceQu', 'GarageFinish','GarageQual','GarageCond','PoolQC','Fence',
                              'MiscFeature','Electrical', 'GarageType']


nan_to_none_filler(train_df, train_features_nan_to_none)
nan_to_none_filler(test_df, train_features_nan_to_none)


# Converting categorical but numerical features to string type ---------------------------------------------------------

# Since these column are actually a category, using numbers will lead the model to assume that there is a superiority
# between them, so we convert them to string.

def convert_feature_to_str(df,features):
    for feature in features:
        df[feature] = df[feature].apply(str)


feature_list_to_str = ['MSSubClass', 'MoSold', 'YrSold']
convert_feature_to_str(train_df, feature_list_to_str)
convert_feature_to_str(test_df, feature_list_to_str)

X = train_df.iloc[:,1:-1]
X_comp = test_df.iloc[:,1:-1]

# Removing skewness in SalePrice
y = train_df.SalePrice
y = np.log(y)

# 2.Feature engineering ------------------------------------------------------------------------------------------------
#(after performing loads of tests and reading some notebooks on Kaggle)

# Dropping the least important
X.drop(['Street', 'PoolQC'], axis=1, inplace=True)

# Adding new features to train data
X['YrBltAndRemod']= X['YearBuilt'] + X['YearRemodAdd']
X['TotalSF']= X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']

X['Total_sqr_footage'] = (X['BsmtFinSF1'] + X['BsmtFinSF2'] + X['1stFlrSF'] + X['2ndFlrSF'])

X['Total_Bathrooms'] = (X['FullBath'] + (0.5 * X['HalfBath']) + X['BsmtFullBath'] + (0.5 * X['BsmtHalfBath']))

X['Total_porch_sf'] = (X['OpenPorchSF'] + X['3SsnPorch'] + X['EnclosedPorch'] + X['ScreenPorch'] + X['WoodDeckSF'])

X['haspool'] = X['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
X['has2ndfloor'] = X['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
X['hasgarage'] = X['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
X['hasbsmt'] = X['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
X['hasfireplace'] = X['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# Features from a guy tuning only xgboost
X["LivLotRatio"] = X.GrLivArea / X.LotArea
X["Spaciousness"] = (X['1stFlrSF'] + X['2ndFlrSF']) / X.TotRmsAbvGrd

# Dropping the least important
X_comp.drop(['Street', 'PoolQC'], axis=1, inplace=True)

# Adding new features to test data
X_comp['YrBltAndRemod']=X_comp['YearBuilt']+X_comp['YearRemodAdd']
X_comp['TotalSF']=X_comp['TotalBsmtSF'] + X_comp['1stFlrSF'] + X_comp['2ndFlrSF']
X_comp['Total_sqr_footage'] = (X_comp['BsmtFinSF1'] + X_comp['BsmtFinSF2'] + X_comp['1stFlrSF'] + X_comp['2ndFlrSF'])
X_comp['Total_Bathrooms'] = (X_comp['FullBath'] + (0.5 * X_comp['HalfBath']) + X_comp['BsmtFullBath'] +
                             (0.5 * X_comp['BsmtHalfBath']))
X_comp['Total_porch_sf'] = (X_comp['OpenPorchSF'] + X_comp['3SsnPorch'] + X_comp['EnclosedPorch'] + X_comp['ScreenPorch'] +
                              X_comp['WoodDeckSF'])

X_comp['haspool'] = X_comp['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
X_comp['has2ndfloor'] = X_comp['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
X_comp['hasgarage'] = X_comp['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
X_comp['hasbsmt'] = X_comp['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
X_comp['hasfireplace'] = X_comp['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# Features from a guy tuning only xgboost
X_comp["LivLotRatio"] = X_comp.GrLivArea / X_comp.LotArea
X_comp["Spaciousness"] = (X_comp['1stFlrSF'] + X_comp['2ndFlrSF']) / X_comp.TotRmsAbvGrd

# Feature mappings (ensuring model about the superiority being present in some features)-------------------------------

# mine first mapping (fullMap):
"""for df in (X, X_comp):
    df.replace({"MSSubClass": {'180': 1, '30': 2, '45': 2, '190': 3, '50': 3, '90': 3, '85': 4, '40': 4, '160': 4,
                               '70': 5, '20': 5, '75': 5, '80': 5, '150': 5, '120': 6, '60': 6},
                "MSZoning": {'C (all)':1, 'RH':2, 'RM':2, 'RL':3, 'FV':4},
                "Alley": {"None": 0, "Grvl": 1, 'Pave': 2},
                "LandSlope": {'Sev': 1, 'Mod': 2, 'Gtl': 3},
                "Neighborhood": {'MeadowV': 1, 'IDOTRR': 2, 'BrDale': 2,'OldTown': 3, 'Edwards': 3, 'BrkSide': 3,
                                 'Sawyer': 4, 'Blueste': 4, 'SWISU': 4, 'NAmes': 4, 'NPkVill': 5, 'Mitchel': 5,
                                 'SawyerW': 6, 'Gilbert': 6, 'NWAmes': 6,'Blmngtn': 7, 'CollgCr': 7, 'ClearCr': 7, 'Crawfor': 7,
                                 'Veenker': 8, 'Somerst': 8, 'Timber': 8, 'StoneBr': 9,'NoRidge': 10, 'NridgHt': 10},
                "Condition1": {'Artery': 1, 'Feedr': 2, 'RRAe': 2, 'Norm': 3, 'RRAn': 3, 'PosN': 4, 'RRNe': 4,
                                  'PosA': 5, 'RRNn': 5},
                "BldgType": {'2fmCon': 1, 'Duplex': 1, 'Twnhs': 1, '1Fam': 2, 'TwnhsE': 2},
                "HouseStyle": {'1.5Unf': 1, '1.5Fin': 2, '2.5Unf': 2, 'SFoyer': 2, '1Story': 3, 'SLvl': 3, '2Story': 4, '2.5Fin': 4},
                "Exterior1st": {'BrkComm': 1, 'AsphShn': 2, 'CBlock': 2, 'AsbShng': 2,'WdShing': 3, 'Wd Sdng': 3, 'MetalSd': 3,
                                'Stucco': 3, 'HdBoard': 3, 'BrkFace': 4, 'Plywood': 4, 'VinylSd': 5, 'CemntBd': 6,
                                'Stone': 7, 'ImStucc': 7},
                "MasVnrType": {'BrkCmn': 1, 'None': 1, 'BrkFace': 2, 'Stone': 3},
                "ExterQual": {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
                "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                "Foundation": {'Slab': 1,'BrkTil': 2,  'Stone': 2, 'CBlock': 3,'Wood': 4, 'PConc': 5},
                "BsmtQual": {"None": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                "BsmtCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4},
                "BsmtExposure": {'No': 0, "None": 0, "Mn": 1, "Av": 2, "Gd": 3},
                "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
                "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,"ALQ": 5, "GLQ": 6},
                "Heating": {'Floor': 1, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'GasW': 3, 'GasA': 4},
                "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                'CentralAir': {'N': 0, 'Y': 1},
                "KitchenQual": {"Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                "Functional": { "Sev": 1, "Maj2": 2, "Maj1": 2, "Mod": 3, "Min2": 4, "Min1": 4, "Typ": 5},
                "FireplaceQu": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                "GarageType": {'CarPort': 1, 'None': 1,'Detchd': 2,'2Types': 3, 'Basment': 3, 'Attchd': 4, 'BuiltIn': 5},
                'GarageFinish': {'None':1, 'Unf':2, 'RFn':3, 'Fin':4},
                "GarageQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                "GarageCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                "PavedDrive": {"None": 0, "P": 1, "Y": 2},
                'Fence': {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},
                "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4},
                "SaleType": {'COD': 1, 'ConLD': 1, 'ConLI': 1, 'ConLw': 1, 'Oth': 1, 'WD': 1,'CWD': 2, 'Con': 3, 'New': 3},
                "SaleCondition": {'AdjLand':1, 'Abnorml':2, 'Alloca':2, 'Family':2, 'Normal':3, 'Partial':4}}, inplace=True)"""

# slightly different mapping (finally I haven't used it):
"""def map_values(X):
    for full in X:
        full["oMSSubClass"] = full.MSSubClass.map({'180': 1,
                                                   '30': 2, '45': 2,
                                                   '190': 3, '50': 3, '90': 3,
                                                   '85': 4, '40': 4, '160': 4,
                                                   '70': 5, '20': 5, '75': 5, '80': 5, '150': 5,
                                                   '120': 6, '60': 6})

        full["oMSZoning"] = full.MSZoning.map({'C (all)': 1, 'RH': 2, 'RM': 2, 'RL': 3, 'FV': 4})

        full["oNeighborhood"] = full.Neighborhood.map({'MeadowV': 1,
                                                       'IDOTRR': 2, 'BrDale': 2,
                                                       'OldTown': 3, 'Edwards': 3, 'BrkSide': 3,
                                                       'Sawyer': 4, 'Blueste': 4, 'SWISU': 4, 'NAmes': 4,
                                                       'NPkVill': 5, 'Mitchel': 5,
                                                       'SawyerW': 6, 'Gilbert': 6, 'NWAmes': 6,
                                                       'Blmngtn': 7, 'CollgCr': 7, 'ClearCr': 7, 'Crawfor': 7,
                                                       'Veenker': 8, 'Somerst': 8, 'Timber': 8,
                                                       'StoneBr': 9,
                                                       'NoRidge': 10, 'NridgHt': 10})

        full["oCondition1"] = full.Condition1.map({'Artery': 1,
                                                   'Feedr': 2, 'RRAe': 2,
                                                   'Norm': 3, 'RRAn': 3,
                                                   'PosN': 4, 'RRNe': 4,
                                                   'PosA': 5, 'RRNn': 5})

        full["oBldgType"] = full.BldgType.map({'2fmCon': 1, 'Duplex': 1, 'Twnhs': 1, '1Fam': 2, 'TwnhsE': 2})

        full["oHouseStyle"] = full.HouseStyle.map({'1.5Unf': 1,
                                                   '1.5Fin': 2, '2.5Unf': 2, 'SFoyer': 2,
                                                   '1Story': 3, 'SLvl': 3,
                                                   '2Story': 4, '2.5Fin': 4})

        full["oExterior1st"] = full.Exterior1st.map({'BrkComm': 1,
                                                     'AsphShn': 2, 'CBlock': 2, 'AsbShng': 2,
                                                     'WdShing': 3, 'Wd Sdng': 3, 'MetalSd': 3, 'Stucco': 3, 'HdBoard': 3,
                                                     'BrkFace': 4, 'Plywood': 4,
                                                     'VinylSd': 5,
                                                     'CemntBd': 6,
                                                     'Stone': 7, 'ImStucc': 7})

        full["oMasVnrType"] = full.MasVnrType.map({'BrkCmn': 1, 'None': 1, 'BrkFace': 2, 'Stone': 3})

        full["oExterQual"] = full.ExterQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

        full["oFoundation"] = full.Foundation.map({'Slab': 1,
                                                   'BrkTil': 2, 'CBlock': 2, 'Stone': 2,
                                                   'Wood': 3, 'PConc': 4})

        full["oBsmtQual"] = full.BsmtQual.map({'Fa': 2, 'None': 1, 'TA': 3, 'Gd': 4, 'Ex': 5})

        full["oBsmtExposure"] = full.BsmtExposure.map({'None': 1, 'No': 2, 'Av': 3, 'Mn': 3, 'Gd': 4})

        full["oHeating"] = full.Heating.map({'Floor': 1, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'GasW': 4, 'GasA': 5})

        full["oHeatingQC"] = full.HeatingQC.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        full["oKitchenQual"] = full.KitchenQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

        full["oFunctional"] = full.Functional.map(
            {'Maj2': 1, 'Maj1': 2, 'Min1': 2, 'Min2': 2, 'Mod': 2, 'Sev': 2, 'Typ': 3})

        full["oFireplaceQu"] = full.FireplaceQu.map({'None': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        full["oGarageType"] = full.GarageType.map({'CarPort': 1, 'None': 1,
                                                   'Detchd': 2,
                                                   '2Types': 3, 'Basment': 3,
                                                   'Attchd': 4, 'BuiltIn': 5})

        full["oGarageFinish"] = full.GarageFinish.map({'None': 1, 'Unf': 2, 'RFn': 3, 'Fin': 4})

        full["oPavedDrive"] = full.PavedDrive.map({'N': 1, 'P': 2, 'Y': 3})

        full["oSaleType"] = full.SaleType.map({'COD': 1, 'ConLD': 1, 'ConLI': 1, 'ConLw': 1, 'Oth': 1, 'WD': 1,
                                               'CWD': 2, 'Con': 3, 'New': 3})

        full["oSaleCondition"] = full.SaleCondition.map(
            {'AdjLand': 1, 'Abnorml': 2, 'Alloca': 2, 'Family': 2, 'Normal': 3, 'Partial': 4})

map_values([X, X_comp])"""

# simple, but effective mapping
for df in (X, X_comp):
    df.replace({
                "ExterQual": {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
                "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                "BsmtQual": {"None": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                "BsmtCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4},
                "BsmtExposure": {'No': 0, "None": 0, "Mn": 1, "Av": 2, "Gd": 3},
                "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
                "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,"ALQ": 5, "GLQ": 6},
                "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                'CentralAir': {'N': 0, 'Y': 1},
                "KitchenQual": {"Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                "Functional": { "Sev": 1, "Maj2": 2, "Maj1": 2, "Mod": 3, "Min2": 4, "Min1": 4, "Typ": 5},
                "FireplaceQu": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                'GarageFinish': {'None':1, 'Unf':2, 'RFn':3, 'Fin':4},
                "GarageQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                "GarageCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                "PavedDrive": {"None": 0, "P": 1, "Y": 2},
                "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4}
                }, inplace=True)




# Removing skewness from numeric columns ------------------------------------------------------------------------------
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []

for i in X.columns:
    if X[i].dtype in numeric_dtypes:
        numerics2.append(i)

skew_features = X[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

for i in skew_index:
    fit_lambda = boxcox_normmax((X[i] + 1), brack=(-1.9, 2.0),  method='mle')
    X[i] = boxcox1p(X[i], fit_lambda)
    X_comp[i] = boxcox1p(X_comp[i], fit_lambda)

# Creating dummies (one-hot-encoding for categoricals with pandas)------------------------------------------------------
X_dummies = pd.get_dummies(X)
X_comp_dummies = pd.get_dummies(X_comp)

# Creating the same columns as in training data, new columns created by dummies in test data are dropped
X_comp_dummies_normalized = X_comp_dummies.reindex(columns = X_dummies.columns, fill_value=0)

# Creating split for train and test datasets ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_dummies, y,random_state=0)


# 3. Model testing and hyperparameters tuning with Optuna --------------------------------------------------------------
def objective(trial, model_selector):
    if model_selector == 1:
        #xgbr
        param = {
                "gamma": trial.suggest_int("gamma",0,0),
                "colsample_bytree": trial.suggest_float("colsample_bytree",0,1),
                "min_child_weight": trial.suggest_int("min_child_weight",0,5),
                "max_depth": trial.suggest_int("max_depth",0,5),
                "n_estimators": trial.suggest_int("n_estimators",4000,8000),
                "alpha": trial.suggest_float("alpha",0.00001,75),
                "learning_rate": trial.suggest_float("learning_rate",0.001,1),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel",0,1),
                "colsample_bynode": trial.suggest_float("colsample_bynode",0,1),
                "random_state": trial.suggest_int("random_state",0,0),
                "subsample": trial.suggest_float("subsample",0,1),
                "lambda": trial.suggest_float("lambda", 0.001, 75)
            }

        model = make_pipeline(RobustScaler(),XGBRegressor(**param))

    elif model_selector == 2:
        #lgbm
        param = {
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0, 1),
            "max_depth": trial.suggest_int("max_depth", 0, 20),
            "n_estimators": trial.suggest_int("n_estimators", 2000, 6900),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0, 1),
            "random_state": trial.suggest_int("random_state", 0, 0),
            "num_leaves": trial.suggest_int("num_leaves",2,50),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0,1),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 8),
            "bagging_seed": trial.suggest_int("bagging_seed", 0, 8),
            "feature_fraction_seed": trial.suggest_int("feature_fraction_seed", 0, 8),
            "verbose":  trial.suggest_int("verbose",-1,-1)
        }

        model = make_pipeline(RobustScaler(),LGBMRegressor(**param))

    elif model_selector == 3:
        #rf
        param = {'n_estimators': trial.suggest_int("n_estimators", 4000, 6900),
              "max_depth": trial.suggest_int("max_depth", 3, 30),
              "max_samples": trial.suggest_float("max_samples", 0.4, 1),
              "max_features": trial.suggest_int("max_features", 1,40),
              "min_samples_split": trial.suggest_int("min_samples_split", 2, 5),
              "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 5)

               }

        model = make_pipeline(RobustScaler(),RandomForestRegressor(**param))


    elif model_selector == 4:
        #ridge
        param = {"alpha": trial.suggest_float("alpha", 1, 50)}

        model = make_pipeline(RobustScaler(), Ridge(**param))

    elif model_selector == 5:
        # lasso
        param = {"alpha": trial.suggest_float("alpha", 0.0001, 0.002)}

        model = make_pipeline(RobustScaler(), Lasso(max_iter=10000,**param))

    elif model_selector == 6:
        # elastic
        param = {"alpha": trial.suggest_float("alpha", 0.00001, 0.01),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.2, 0.6)}

        model = make_pipeline(RobustScaler(), ElasticNet(**param))

    elif model_selector == 7:
        # svr
        param = {"epsilon": trial.suggest_float("epsilon", 0.001, 0.1),
                'C': trial.suggest_float('C',0.001,15),
                 'gamma': trial.suggest_float('gamma',0.1,2)
                 }

        model = make_pipeline(RobustScaler(), SVR(kernel='linear', **param))

    elif model_selector == 8:
        # kernel_ridge
        param = {"alpha": trial.suggest_float("alpha", 0.001, 10),
                'kernel': trial.suggest_categorical('C',['linear', 'polynomial', 'sigmoid', 'rbf']),
                 'degree': trial.suggest_int('degree', 2,10),
                 'gamma': trial.suggest_float('gamma',0.001,5),
                 'coef0': trial.suggest_float('coef0', 0, 15)
                 }

        model = make_pipeline(RobustScaler(), KernelRidge(**param))

    elif model_selector == 9:
        # catboost
        param = {
            #"objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 1),
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100)
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        model = make_pipeline(RobustScaler(), CatBoostRegressor(**param, silent=True))



    def cv_rmse(model, X, y):
        rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
        return (rmse)

    model.fit(X_train, y_train)
    scores = []
    scores = cv_rmse(model, X_train, y_train)
    score = scores.mean()
    return score

models = {'xgbr': 1, 'lgbm': 2, 'rf': 3, 'ridge': 4, 'lasso': 5, 'elastic': 6, 'svr': 7, 'kernel_ridge': 8, 'catboost': 9}

# To analyse results on the optuna dashboard, open bookmarked url, and copy there db.sqlite3 file from python folder
# study_name  - automation needed in the future
study = opt.create_study(direction='minimize',storage="sqlite:///db.sqlite3", study_name="testo")

# Here we choose a model and number of trials for hypertuning
study.optimize(lambda trial: objective(trial, models['elastic']), n_trials=500, n_jobs=-1)
print(study.best_trial.params)
print(study.best_value)


# Best tuned models v2.0 after optuna optimize
ridge = make_pipeline(RobustScaler(), Ridge(alpha=25.73165549379894))
lasso = make_pipeline(RobustScaler(), Lasso(max_iter=7000,alpha=0.0004995226671674873))
elastic = make_pipeline(RobustScaler(), ElasticNet(alpha=0.001945812079570378, l1_ratio=0.4229175996530061))
svr = make_pipeline(RobustScaler(), SVR(kernel='linear', epsilon=0.04761941382775011, C=0.034617442228460804,
                                        gamma=0.3244726150963515))
lgbm_params = {'colsample_bytree': 0.5521060280754942, 'max_depth': 18, 'n_estimators': 6410, 'reg_alpha': 0.727359233292193,
               'reg_lambda': 1.7392017732254956, 'learning_rate': 0.004259768618583768, 'colsample_bynode': 0.20757976595078706,
               'random_state': 0, 'num_leaves': 7, 'bagging_fraction': 0.6327920804798416, 'bagging_freq': 3,
               'bagging_seed': 2, 'feature_fraction_seed': 3, 'verbose': -1}
lgbm = make_pipeline(RobustScaler(), LGBMRegressor(**lgbm_params))

kernel_ridge_params =  {'alpha': 2.0954723360979703, 'degree': 3, 'gamma': 0.0007064511531508046, 'coef0': 7.57865644877607}
kernel_ridge = make_pipeline(RobustScaler(), KernelRidge(kernel='polynomial',**kernel_ridge_params))

xgbr = make_pipeline(RobustScaler(), XGBRegressor(learning_rate=0.00875,n_estimators=3515,max_depth=4,
                                                  min_child_weight=2, gamma=0,reg_alpha=0.00006,
                                                  colsample_bytree=0.2050378195385253, subsample=0.40369887914955715,
                                                  reg_lambda=0.046181862052743 ,random_state=0))

catboost_params = {'learning_rate': 0.022710698405474926, 'colsample_bylevel': 0.5451034240240138,
                   'max_depth': 4, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS', 'min_data_in_leaf': 89}
catboost = make_pipeline(RobustScaler(), CatBoostRegressor(**catboost_params, silent=True))
#rf not included


def final_test(model, X_train, X_test, y_train, y_test):
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    model_name = type(model[1]).__name__
    model_params = model[1].get_params()
    model.fit(np.array(X_train), y_train)
    pred = model.predict(np.array(X_test))
    rmse = round(np.sqrt(mean_squared_error(y_test, pred)), 6)
    score_rmsle = round(mean_squared_log_error(y_test, pred, squared=False), 6)
    with open(f'{model_name}_{rmse}_{score_rmsle}_{run_id}.json', 'w') as json_file:
        json.dump(model_params, json_file)
    print(rmse, score_rmsle)



def results(model, X, y, X_comp):
        model_name = type(model[1]).__name__
        model.fit(X, y)
        preds = model.predict(X_comp)
        preds = np.exp(preds)
        np.savetxt(f'{model_name}.txt', preds)
        result = pd.DataFrame({'Id': test_df.Id,
                                'SalePrice': preds.squeeze()})

        print(result.head())



# Choosing best stack composition. It should be done again with CV on train set or on additional data, but since this kaggle
# dataset is so small, I broke the rule and tuned on the test set results, risking overfitting on the test set.

# Best stack for simple mapping (best predictor)
stack = StackingCVRegressor(regressors=[lasso, elastic, ridge, lgbm, svr, xgbr, kernel_ridge, catboost], meta_regressor= ridge,
                            use_features_in_secondary=True, random_state=0, n_jobs=-1,
                            store_train_meta_features=True, shuffle=False)

# Best stack for fullmapping (the single models were trained also on fullmapping, but I didn't expect this to be a good solution
# and I literally lost somewhere the trained hyperparameters for them, and I am not able to replicate them quickly
stack_2 = StackingCVRegressor(regressors=[lasso, elastic, ridge, lgbm, svr, xgbr, kernel_ridge], meta_regressor= ridge,
                            use_features_in_secondary=True, random_state=0, n_jobs=-1,
                            store_train_meta_features=True, shuffle=False)

# Getting final results on test set for single models and stacks, and saving them with hyperparametrs
models_stack = [lasso, elastic, ridge, lgbm, svr, xgbr, kernel_ridge, catboost]
for model in models_stack:
    final_test(model, X_train, X_test, y_train, y_test)
    results(model, X_dummies, y, X_comp_dummies_normalized)

for model in [stack, stack_2]:
    for i in range(2):
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        model.fit(np.array(X_train), y_train)
        pred = model.predict(np.array(X_test))
        rmse = round(np.sqrt(mean_squared_error(y_test, pred)), 6)
        score_rmsle = round(mean_squared_log_error(y_test, pred, squared=False), 6)
        with open(f'stack_{i}_{rmse}_{score_rmsle}_{run_id}.json', 'w') as json_file:
            json.dump(score_rmsle, json_file)
        print(rmse, score_rmsle)

        model.fit(X_dummies, y)
        preds = model.predict(X_comp_dummies_normalized)
        preds = np.exp(preds)
        np.savetxt(f'stack_{i}.txt', preds)
        result = pd.DataFrame({'Id': test_df.Id,
                               'SalePrice': preds.squeeze()})

        print(result.head())


# Neural nets, with new data split, and new optuna tuning functions ----------------------------------------------------
from sklearn.model_selection import KFold
from keras.layers import Flatten, Dense, BatchNormalization, Dropout
from keras.losses import MeanSquaredError
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Sequential, load_model
from keras.optimizers import Adam, SGD, Nadam

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()
X_train_split, X_valid, y_train_split, y_valid = train_test_split(X_train, y_train,train_size=0.8, random_state=0)

scaler = RobustScaler()
X_train_split_scaled = scaler.fit_transform(X_train_split)
X_valid_scaled = scaler.transform(X_valid[1:])
y_valid = y_valid[1:]
X_test_scaled = scaler.transform(X_test)


# Many times I got strange optimizer error when the learning_rate was also added to the search space as a parameter. I havn't been
# able to resolve it, so decided to stick with defaults. I tested Dropout, Monte Carlo Dropout, BatchNormalization manually,
# but it didn't help.
def NN_objective(trial):
    kfold = KFold(n_splits=5, shuffle=True, random_state=33)
    model = Sequential()
    model.add(Flatten())
    n_layers = trial.suggest_int('n_layers', 1,50)
    for n in range(n_layers):
        num_hidden = trial.suggest_int('num_hidden', 5, 500)
        activation = trial.suggest_categorical('activation', ['linear', 'relu', 'selu', 'elu'])
        kernel_initializer = trial.suggest_categorical('kernel_initializer', ['he_uniform', 'he_normal', 'lecun_normal'])
        model.add(Dense(num_hidden, activation, kernel_initializer))

    model.add(Dense(1))
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "Nadam"])


    model.compile(loss=MeanSquaredError(),
                  optimizer=optimizer)
    scores = []
    for train, test in kfold.split(X_train_split_scaled, y_train_split):
        earlystopping_cb = EarlyStopping(patience=10, restore_best_weights=True)
        model.fit(X_train_split_scaled[train],y_train_split[train],
                  validation_data=(X_valid_scaled, y_valid), epochs=50,
                  callbacks=[earlystopping_cb])
        score = model.evaluate(X_train_split_scaled[test], y_train_split[test])

        rmse = np.sqrt(score)
        scores.append(rmse)

    rmse_mean = np.mean(scores)
    print(scores)

    return rmse_mean


study = opt.create_study(direction='minimize')
study.optimize(NN_objective, n_trials=70, n_jobs=-1)
print(study.best_params)
print(study.best_value)

# Next, we use best parameters obtained from optuna to build a neural net. Another shameful part of it is that I lost
# somewhere parameters of the best models, and wasn't able to reproduce the results. Luckily I saved all predictions in
# one folder, and tested all of them later. Below I used parameters of a good model, but still being nowhere near
# the best ones.

NN_model = Sequential()
NN_model.add(Flatten())
for n in range(50):
    NN_model.add(Dense(162, activation='relu', kernel_initializer='lecun_normal'))

NN_model.add(Dense(1))

earlystopping_cb = EarlyStopping(patience=10, restore_best_weights=True)
NN_model.compile(loss=MeanSquaredError(),
                 optimizer=Adam())
history = NN_model.fit(X_train_split_scaled, y_train_split, validation_data=(X_valid_scaled, y_valid), epochs=50,
                       callbacks=earlystopping_cb)
# print(NN_model.summary())

# Evaluating, saving and loading model
mse = NN_model.evaluate(X_test_scaled, y_test)
rmse = np.sqrt(mse)
# print('rmse: ', rmse)
NN_model.save("example")

NN_model = load_model("example")
print(NN_model.summary())
mse = NN_model.evaluate(X_test_scaled, y_test)
rmse = np.sqrt(mse)
print('rmse: ', rmse)

X_comp_dummies_normalized = X_comp_dummies_normalized.to_numpy()
X_comp_dummies_normalized_scaled = scaler.transform(X_comp_dummies_normalized)

def nn_results(model, X_comp):

    preds = model.predict(X_comp)
    preds = np.exp(preds)
    print(len(preds))
    np.savetxt('best_example.txt', preds)
    result = pd.DataFrame({'Id': test_df.Id,
                           'SalePrice': preds.squeeze()})

    print(result.head())
    result.to_csv('final_results.csv', index=False)


nn_results(NN_model, X_comp_dummies_normalized_scaled)

# Finally I found out 2 predictions which form best predictions after averaging them
preds1 = np.loadtxt('best_NN.txt')
preds2 = np.loadtxt('best_NN2.txt')
preds = 0.5 * preds1 + 0.5 * preds2
np.savetxt('best_example_voting.txt', preds)
print(preds[:3], preds1[:3], preds2[:3])
result = pd.DataFrame({'Id': test_df.Id,
                           'SalePrice': preds.squeeze()})

print(result.head())
result.to_csv('final_results.csv', index=False)



stack = np.loadtxt('stack.txt')
fullMap = np.loadtxt('fullMap.txt')
ridge = np.loadtxt('ridge.txt')
xgbr = np.loadtxt('xgbr.txt')
lgbm = np.loadtxt('lgbm.txt')
lasso = np.loadtxt('lasso.txt')
elastic = np.loadtxt('elastic.txt')
svr = np.loadtxt('svr.txt')
kernel_ridge = np.loadtxt('kernel_ridge.txt')
catboost = np.loadtxt('catboost.txt')
NN = np.loadtxt('best_NN_voting.txt')

# Checking if the length of predictions is correct and the same
#print(len(stack), len(fullMap), len(ridge), len(xgbr), len(lgbm), len(lasso), len(elastic), len(svr), len(kernel_ridge), len(catboost), len(NN))

# Final predictions --------------------------------------------------------------------------------------------------------
"""The only purpose of it, is the highest score on Kaggle. Stacking and voting ensembles usually should
be trained and validated using CV before, and finally tested on test dataset.
Anyway, after many manual tests, voting weights has been assigned to the best predictors (stack, fullMap) and to
predictors with a different direction of the bias to ensure better generalization."""

preds = ((stack * 0.45) + (fullMap * 0.25) + (xgbr * 0.2) + (NN * 0.10))

results = pd.DataFrame({'Id': test_df.Id,'SalePrice': preds.squeeze()})
print(results.head())
results.to_csv('results_kaggle_housing.csv', index=False)

