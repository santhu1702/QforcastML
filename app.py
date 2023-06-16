# import pyodbc
import pandas as pd
import pyodbc as po
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from flask import Flask, abort, request
import traceback
import pickle
import threading
import constraints
import commonmethods
import datetime
import os

app = Flask(__name__)

conn = po.connect(
    'DRIVER=' + constraints.driver + ';SERVER=tcp:' + constraints.server + ';PORT=' + constraints.port + ';DATABASE=' + constraints.database + ';UID=' + constraints.username + ';PWD=' + constraints.password)

monthDic = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
            'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
 

# algorithm = {'Linear Regression': LinearRegression(),
#              'DecisionTree': DecisionTreeRegressor(),
#              'RandomForest': RandomForestRegressor(),
#              'KNeighbors': KNeighborsRegressor(),
#              'SVM': SVR(), 'XGBoost': XGBRegressor(),
#              'AdaBoost': AdaBoostRegressor()}

algorithm = {'RandomForest': RandomForestRegressor()}
###########################################################################################################################


@app.route('/', methods=["POST", "GET"])
def user_login():
    return "Hosted successfully"

###########################################################################################################################


@app.route('/train', methods=["POST", "GET"])
def train():
    threading.Thread(target=train_model).start()
    #train_model()
    commonmethods.sendMail("Model is getting trained")
    return "Model is getting trained"
###########################################################################################################################


def train_model():
    try:
        # df = sqlCall('All', 'All', 'All', 'All', 'All', 'All', 'All')
        df = sqlCallActualSalesData()
        # print("sqlcall")
        commonmethods.sendMail("Sql call has been done" +
                               str(datetime.datetime.now()))
        data = pd.melt(df,
                       id_vars=['ProductName', 'CategoryName', 'FiscalYear', 'CountryName', 'CustomerName', 'BusinessName',
                                'DimProductId', 'DimCountryId', 'DimProductSubCategoryId', 'DimLineofBusinessId',
                                'DimCustomerId'],
                       value_vars=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        data["FiscalYear"] = data["FiscalYear"].astype("int")
        data["DimProductId"] = data["DimProductId"].astype("int")
        data["DimCountryId"] = data["DimCountryId"].astype("int")
        data["DimProductSubCategoryId"] = data["DimProductSubCategoryId"].astype(
            "int")
        data["DimLineofBusinessId"] = data["DimLineofBusinessId"].astype("int")
        data["DimCustomerId"] = data["DimCustomerId"].astype("int")

        data.rename(columns={"variable": "Month",
                    "value": "budgetvalue"}, inplace=True)

        data['budgetvalue'] = data['budgetvalue'].replace(np.nan, 0)

        data.drop(['ProductName', 'CategoryName', 'CountryName', 'CustomerName', 'BusinessName'],
                  axis=1, inplace=True)

        data['Month'] = data['Month'].map(monthDic)
        df_train, df_test = train_test_split(
            data, train_size=0.7, random_state=100)
        # print("y_train")

        y_train = df_train['budgetvalue'] # type: ignore
        # print("X_train")

        X_train = df_train.drop(['budgetvalue'], axis=1)
        Scaler = MinMaxScaler()

        X_train[X_train.columns] = Scaler.fit_transform(
            X_train[X_train.columns])
        y_test = df_test.pop('budgetvalue')
        X_test = df_test

        X_test[X_test.columns] = Scaler.transform(X_test[X_test.columns])
        pickle.dump(Scaler, open("scaler.pkl", 'wb'))
        # print("qforecastmodel trainscore started")

        trainscore = qforecastmodel(X_train, y_train)
        # print("qforecastmodel trainscore ended")

        trainscore.sort_index()

        # print("qforecastmodel testscore started")
        testscore = qforecastmodel(X_test, y_test)
        # print("qforecastmodel testscore ended")

        suitablemodeltest = suitable_model(testscore)
        suitablemodeltrain = suitable_model(trainscore)
        # print("qforecastmodel finding algorithm started")

        commonmethods.sendMail("finding algorithm" +
                               str(datetime.datetime.now()))
        # classifier = find_algorithm(suitablemodeltrain, suitablemodeltest)
        classifier = RandomForestRegressor()
        commonmethods.sendMail("selected algorithm" +
                               str(datetime.datetime.now()))
        # print("classifier",classifierclassifier)
        commonmethods.sendMail("classifier started" +
                               str(datetime.datetime.now()))
        qfmodel = classifier.fit(X_train, y_train)
        commonmethods.sendMail("classifier end" +
                               str(datetime.datetime.now()))
        pickle.dump(qfmodel, open('model.pkl', 'wb'))
        commonmethods.sendMail("model generated" +
                               str(datetime.datetime.now()))
        commonmethods.sendMail("model trained" + str(datetime.datetime.now()))
        return "model trained"
    except Exception as e:
        # print(traceback.format_exc())
        commonmethods.sendMail(
            str(traceback.format_exc()) + str(datetime.datetime.now()))
        return str(e)

##########################################################################################################################


@app.route('/get_forecast', methods=["GET"])
def get_forecast():
    try:
        # if not os.path.isfile(os.path.join(os.getcwd(), 'model.pkl')):
        #     abort(404, "The file   does not exist.")

        qfmodel = pickle.load(open('model.pkl', 'rb'))
        
        validation_data = sqlCall(
            '2022', 'All', 'All', 'All', 'All', 'All', 'All',25,1)
        validation_data["FiscalYear"] = validation_data["FiscalYear"].astype(
            "int")
        validation_data["DimProductId"] = validation_data["DimProductId"].astype(
            "int")
        validation_data["DimCountryId"] = validation_data["DimCountryId"].astype(
            "int")
        validation_data["DimProductSubCategoryId"] = validation_data["DimProductSubCategoryId"].astype(
            "int")
        validation_data["DimLineofBusinessId"] = validation_data["DimLineofBusinessId"].astype(
            "int")
        validation_data["DimCustomerId"] = validation_data["DimCustomerId"].astype(
            "int")
        # unpivot the validation data
        validation_data = pd.melt(validation_data,
                                  id_vars=['ProductName', 'CategoryName', 'FiscalYear', 'CountryName', 'CustomerName',
                                           'BusinessName', 'DimProductId', 'DimCountryId', 'DimProductSubCategoryId',
                                           'DimLineofBusinessId', 'DimCustomerId'],
                                  value_vars=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                                              'Nov',
                                              'Dec'])
        validation_data.drop('value', axis=1, inplace=True)
        validation_data.rename(columns={'variable': 'Month'}, inplace=True)
        data_val = validation_data.drop(['ProductName', 'CategoryName', 'CountryName', 'CustomerName',
                                         'BusinessName'], axis=1)
        data_val['Month'] = data_val['Month'].map(monthDic)
        # input from API
        predicted_year = request.args['year']
        # we have to declare as input variable
        data_val["FiscalYear"] = predicted_year
        Scaler = pickle.load(open('scaler.pkl', 'rb'))
        data_val[data_val.columns] = Scaler.transform(
            data_val[data_val.columns])
        y_validation = qfmodel.predict(data_val)
        validation_data['Pred_val'] = np.round_(y_validation, decimals=3)
        validation_data['FiscalYear'] = predicted_year
        val_data = validation_data.pivot(index=['ProductName', 'CategoryName', 'FiscalYear', 'CountryName',
                                                'CustomerName', 'BusinessName', 'DimProductId', 'DimCountryId',
                                                'DimProductSubCategoryId', 'DimLineofBusinessId', 'DimCustomerId'],
                                         columns='Month', values='Pred_val')
        # val_data.to_csv("predicted.csv")
        # to json
        # to json
        # val_data.to_json("predicted.json")
        return val_data.to_csv()
    except Exception as e:
        print(traceback.format_exc())
       # commonmethods.sendMail(
        #    "model trained "+str(datetime.datetime.now()) + " error " + str(traceback.format_exc()))
        abort(404, str(e))


def eval_metrics(y, y_pred):
    MSE = metrics.mean_squared_error(y, y_pred)
    MAE = metrics.mean_absolute_error(y, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y, y_pred))
    r2 = metrics.r2_score(y, y_pred)
    return MSE, MAE, RMSE, r2


# def qforecastmodel(X, y):
#     col_name = ['Algorithm', 'MSE', 'MAE', 'RMSE', 'R2 Score']

#     result = pd.DataFrame(columns=col_name)
#     i = 0

#     for name, classifier in algorithm.items():
#         classifier = classifier
#         qfmodel = classifier.fit(X, y)
#         y_train_pred = qfmodel.predict(X)
#         mse, mae, rmse, r2 = eval_metrics(y, y_train_pred)
#         result.loc[i] = [name, mse.round(2), mae.round(
#             2), rmse.round(2), r2.round(2)]
#         i += 1
#     result.set_index(['Algorithm'], inplace=True)
#     return result
def qforecastmodel(X, y):
    col_name = ['Algorithm', 'MSE', 'MAE', 'RMSE', 'R2 Score']
    algorithms = algorithm.items()

    result = pd.DataFrame([[
        name, round(mse, 2), round(mae, 2), round(rmse, 2), round(r2, 2)
    ] for name, classifier in algorithms for mse, mae, rmse, r2 in [eval_metrics(y, classifier.fit(X, y).predict(X))]
    ], columns=col_name).set_index('Algorithm')

    return result


def sqlCallActualSalesData():
    return pd.read_sql("EXEC USP_ActualSalesData ", conn)


def sqlCall(fiscalYear, LOB, product, productsubcategory, customer, country, state,UserID,usertypeid):
    fiscal_year = fiscalYear
    dim_line_of_business_id = LOB
    dim_product_id = product
    dim_product_subcategory_id = productsubcategory
    dim_customer_id = customer
    dim_country_id = country
    dim_state_id = state
    UserID = int(UserID)
    usertypeid = int(usertypeid)
    params = (
        fiscal_year, dim_line_of_business_id, dim_product_id, dim_product_subcategory_id, dim_customer_id,
        dim_country_id,
        dim_state_id,UserID,usertypeid)

    return pd.read_sql(
        "{CALL USP_NewGETFact_ActualSaleswithUserID (?, ?, ?, ?, ?, ?, ? , ?  ,?)}", conn, params=params)


################################################################
def suitable_model(trainscore):
    l = []
    m = []
    for i in trainscore['RMSE']:
        for j in trainscore['R2 Score']:
            if (i > 0) & (j != 1):
                l.append(i)
                m.append(j)
    low_rmse = min(l)
    high_r2score = max(m)
    if (trainscore[trainscore['RMSE'] == low_rmse].index.values) == (trainscore[trainscore['R2 Score'] == high_r2score].index.values):
        strain_algo = trainscore[trainscore['RMSE'] == low_rmse].index.values
        for i in strain_algo:
            strain = i
        return strain
    else:
        strain_algo = trainscore[trainscore['R2 Score']
                                 == high_r2score].index.values
        for i in strain_algo:
            strain = i
        return strain


def find_algorithm(suitablemodeltest, suitablemodeltrain):
    if suitablemodeltest == suitablemodeltrain:
        for k, v in algorithm.items():
            if k == suitablemodeltest:
                classifier = v
        return classifier


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
