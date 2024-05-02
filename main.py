import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def preprocess_data(filename):
    df=pd.read_csv(filename, 
               names=["Id", "MSSubclass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley", 
                      "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", 
                      "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond", 
                      "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", 
                      "MasVnrType", "MasVnrArea", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", 
                      "BsmtExposure", "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", 
                      "TotalBsmtSF", "Heating", "HeatingQC", "CentralAir", "Electrical", "1stFlrSF", "2ndFlrSF", 
                      "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", 
                      "KitchenAbvGr", "KitchenQual", "TotRmsAbvGrd", "Functional", "Fireplaces", "FireplaceQu", "GarageType", 
                      "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond", "PavedDrive", 
                      "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "PoolQC", "Fence", 
                      "MiscFeature", "MiscVal", "MoSold", "YrSold", "SaleType", "SaleCondition", "SalePrice"])

    le = preprocessing.LabelEncoder()
    for col in df.columns:
       df[col] = le.fit_transform(df[col])

    x = df[(list(df.columns[:-1]))]
    y = df['SalePrice']

    return x,y


def SVM(x_train, x_test, y_train, y_test):
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("MAE: ", mean_absolute_error(y_test, y_pred)
    print("MAPE: ", mean_absolute_percentage_error(y_test, y_pred))

def main():
    x_train, y_train = preprocess_data("train.csv")
    x_test, y_test = preprocess_data("test.csv")
    SVM(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()



