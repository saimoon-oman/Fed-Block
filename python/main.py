import uvicorn
import pickle
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
# from model.main import run
from model.server import Server
from model.client import Client
from sklearn.utils import shuffle
import pandas as pd
import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import requests
import ast
from sklearn.preprocessing import StandardScaler

# **************************************
# data cleaning and plots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# %matplotlib inline

# sklearn: data preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# sklearn: train model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report

# sklearn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# **************************************


server = Server()
X_test_list = []
y_test_list = []

class Input(BaseModel):
    s_hash: list[str]
    x: int

# Initializing the fast API server
app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/aggregate/")
async def aggregate(request:Request):
#**************************FRAMINGHAM_DATASET******************************
    # projectId = "2HWWSM26k2fOYJGFAiAxCsF2TRF"
    # projectSecret = "ab5fa9048e3166641c0a2726b6351230"
    # endpoint = "https://ipfs.infura.io:5001"
    # # print(await request.json())

    
    # res = await request.json()      # Retrieve JSON data from the HTTP request
    # print("result")
    # print(res)

    # clients = list()
    # # Process each item in the received data
    # for i in res:
    #     # Prepare parameters for IPFS request
    #     params = {
    #         'arg': i
    #     }

    #     # Send a request to IPFS to retrieve data
    #     response2 = requests.post(endpoint + '/api/v0/cat', params=params, auth=(projectId, projectSecret))
    #     # print(response2)
    #     # print(response2.text)
        

    #     # Process the data from IPFS and create a list of clients
    #     with open(response2.text) as f:
    #         print(f"Value of f: {f}")
    #         for line in f:
    #             x = line.split(",")
    #             # print(x)
    #             client=list()
    #             coef=list()
    #             intercept=list()
    #             classes=list()
    #             x[0]=x[0][2:]

    #             # Process and extract coefficients, intercept, and classes
    #             for i in range(13):
    #                 coef.append(float(x[i][1:]))
    #             x[13] = x[13][1:]
    #             x[13] = x[13][:-1]
    #             coef.append(float(x[13]))
    #             x[14]=x[14][2:]
    #             x[14]=x[14][:-1]
    #             intercept.append(float(x[14]))
    #             classes.append(int(x[15][-1]))
    #             classes.append(int(x[16][1]))
    #             client.append(coef)
    #             client.append(intercept)
    #             client.append(classes)
    #             print(client)
    #             clients.append(client)

    # # Initialize a server and update its model using client data
    # # server = Server()
    # server.update_model(clients)
    # # Load a CSV dataset, process, and prepare it for testing
    # # df = pd.read_csv('model/framingham_test.csv')
    # # # print(df.head())
    # # df = df.dropna()
    # # df.fillna(method='bfill', inplace=True)
    # # df = shuffle(df)
    # # y = df["TenYearCHD"]
    # # X = df.drop(columns=['TenYearCHD', 'education'], axis=1)

    # accuracy = list()
    # print(len(clients))
    # # Perform model testing on data segments
    # for i in range(len(clients)):
    #     acc = server.test(X_test_list[i], y_test_list[i])
    #     accuracy.append(int(acc*(10**5)))
 
    # print("*************************************************************************")

    # print(accuracy)
    # # acc = server.test(X, y)

    # server_grads=list()
    # server_grads.append(server.model.coef_[0].tolist())
    # server_grads.append(server.model.intercept_.tolist())
    # server_grads.append(server.model.classes_.tolist())
    # print('Server Grads: ', server_grads)

    # server_file = open("server.txt", "w+")

    # # Saving the 2D array in a text file
    # server_content = str(server_grads)

    # server_file.write(server_content)
    # server_file.close()
    # file = {
    #     'server_file': 'server.txt',
    # }

    # ### ADD FILE TO IPFS AND SAVE THE HASH ###
    # server_response = requests.post(endpoint + '/api/v0/add', files=file, auth=(projectId, projectSecret))
    # print("Server Response:",server_response)
    # server_hash = server_response.text.split(",")[1].split(":")[1].replace('"', '')
    # print("Server Hash:", server_hash)
    # return {
    #     "data": {
    #         "accuracy": accuracy,
    #         "hash": server_hash
    #     }
    # }

#**************************UNSW_NB15_DATASET******************************
    projectId = "2HWWSM26k2fOYJGFAiAxCsF2TRF"
    projectSecret = "ab5fa9048e3166641c0a2726b6351230"
    endpoint = "https://ipfs.infura.io:5001"
    # print(await request.json())

    
    res = await request.json()      # Retrieve JSON data from the HTTP request
    # print("result")
    # print(res)

    clients = list()
    # Process each item in the received data
    for i in res:
        # Prepare parameters for IPFS request
        params = {
            'arg': i
        }

        # Send a request to IPFS to retrieve data
        response2 = requests.post(endpoint + '/api/v0/cat', params=params, auth=(projectId, projectSecret))
        # print(response2)
        # print(response2.text)
        

        # Process the data from IPFS and create a list of clients
        with open(response2.text) as f:
            print(f"Value of f: {f}")
            for line in f:
                print("Line: ", line)
                print("Type of Line: ", type(line))
                
                client=list()
                # Safely evaluate the string into a Python object
                data_list = ast.literal_eval(line)
                
                # Flatten and reshape the nested list
                coef = data_list[0]
                intercept = data_list[1]
                classes = data_list[2]

                # x = line.split(",")
                # print("X[0]: ", x[0])
                # # print(x)
                # # client=list()
                # coef=list()
                # intercept=list()
                # classes=list()
                # x[0]=x[0][2:]

                # # Process and extract coefficients, intercept, and classes
                # for i in range(13):
                #     coef.append(float(x[i][1:]))
                # x[13] = x[13][1:]
                # x[13] = x[13][:-1]
                # coef.append(float(x[13]))
                # x[14]=x[14][2:]
                # x[14]=x[14][:-1]
                # intercept.append(float(x[14]))
                # classes.append(int(x[15][-1]))
                # classes.append(int(x[16][1]))
                client.append(coef)
                client.append(intercept)
                client.append(classes)
                print(client)
                clients.append(client)

    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    # Initialize a server and update its model using client data
    # server = Server()
    server.update_model(clients)
    # Load a CSV dataset, process, and prepare it for testing
    # df = pd.read_csv('model/framingham_test.csv')
    # # print(df.head())
    # df = df.dropna()
    # df.fillna(method='bfill', inplace=True)
    # df = shuffle(df)
    # y = df["TenYearCHD"]
    # X = df.drop(columns=['TenYearCHD', 'education'], axis=1)

    accuracy = list()
    print(len(clients))
    # Perform model testing on data segments
    for i in range(len(clients)):
        acc = server.test(X_test_list[i], y_test_list[i])
        accuracy.append(int(acc*(10**5)))
 
    print("*************************************************************************")

    print(accuracy)
    # acc = server.test(X, y)

    server_grads=list()
    server_grads.append(server.model.coef_.tolist())
    server_grads.append(server.model.intercept_.tolist())
    server_grads.append(server.model.classes_.tolist())
    print('Server Grads: ', server_grads)

    server_file = open("server.txt", "w+")

    # Saving the 2D array in a text file
    server_content = str(server_grads)

    server_file.write(server_content)
    server_file.close()
    file = {
        'server_file': 'server.txt',
    }

    ### ADD FILE TO IPFS AND SAVE THE HASH ###
    server_response = requests.post(endpoint + '/api/v0/add', files=file, auth=(projectId, projectSecret))
    print("Server Response:",server_response)
    server_hash = server_response.text.split(",")[1].split(":")[1].replace('"', '')
    print("Server Hash:", server_hash)
    return {
        "data": {
            "accuracy": accuracy,
            "hash": server_hash
        }
    }

# Setting up server model with random weight
@app.get("/initiallySetServerWeights/")
async def initiallySetServerWeights(): 
#**************************FRAMINGHAM_DATASET******************************
    # model = LogisticRegression()
    # df = pd.read_csv('model/framingham.csv')
    # df = df.dropna()
    # df.fillna(method='bfill', inplace=True)
    # data = df[28:30]

    # y = data["TenYearCHD"]
    # X = data.drop(columns=['TenYearCHD', 'education'], axis=1)
    # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X)

    # model.fit(X_train, y)

    # # print("Model Coefficient")
    # # print(model.coef_)
    # # print(model.intercept_)
    # # print(model.classes_)

    # # server.model.coef_ = np.array(model.coef_)
    # # server.model.intercept_ = model.intercept_
    # # server.model.classes_ = np.array(list(model.classes_))

    # # print(server.model.coef_)
    # # print(server.model.intercept_)
    # # print(server.model.classes_)

    # projectId = "2HWWSM26k2fOYJGFAiAxCsF2TRF"
    # projectSecret = "ab5fa9048e3166641c0a2726b6351230"
    # endpoint = "https://ipfs.infura.io:5001"
    
    # server_grads=list()
    # server_grads.append(model.coef_[0].tolist())
    # server_grads.append(model.intercept_.tolist())
    # server_grads.append(model.classes_.tolist())
    # print('Server Grads: ', server_grads)

    # server_file = open("server.txt", "w+")

    # # Saving the 2D array in a text file
    # server_content = str(server_grads)

    # server_file.write(server_content)
    # server_file.close()
    # file = {
    #     'server_file': 'server.txt',
    # }

    # ### ADD FILE TO IPFS AND SAVE THE HASH ###
    # server_response = requests.post(endpoint + '/api/v0/add', files=file, auth=(projectId, projectSecret))
    # print("Server Response:",server_response)
    # server_hash = server_response.text.split(",")[1].split(":")[1].replace('"', '')
    # print("Server Hash:", server_hash)
    # return {
    #     "hash": server_hash
    # }

#**************************UNSW_NB15_DATASET******************************
    # Load data
    initial_data = pd.read_csv('model/UNSW_NB15_training-set.csv')
    # # Shuffle the rows
    # initial_data = initial_data.sample(frac=1).reset_index(drop=True)

    # information of the data: 583 data points, 10 features' columns and 1 target column
    initial_data = initial_data.drop(axis=1, columns=['id'])
    initial_data = initial_data.drop(axis=1, columns=['proto'])
    initial_data = initial_data.drop(axis=1, columns=['service'])
    initial_data = initial_data.drop(axis=1, columns=['state'])
    # initial_data.info()

    # Discard the rows with missing values
    data_to_use = initial_data.dropna()
    data_to_use = data_to_use[47910: 48000]

    # Shape of the data: we could see that the number of rows remains the same as no null values were reported
    # print(data_to_use.shape)

    X = data_to_use.drop(axis=1, columns=['attack_cat']) # X is a dataframe
    X_train = X.drop(axis=1, columns=['label'])

    y1_train = data_to_use['attack_cat'].values # y is an array
    y2_train = data_to_use['label'].values

    # determine categorical and numerical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns

    # define the transformation methods for the columns
    t = [('ohe', OneHotEncoder(drop='first'), categorical_cols),
        ('scale', StandardScaler(), numerical_cols)]

    col_trans = ColumnTransformer(transformers=t)

    # fit the transformation on training data
    col_trans.fit(X_train)

    X_train_transform = col_trans.transform(X_train)
    print("Shape of X_train_transform: ", X_train_transform.shape)

    # Define a LabelEncoder() transformation method and fit on y1_train
    target_trans = LabelEncoder()
    target_trans.fit(y1_train)

    # apply transformation method on y1_train and y1_test
    y1_train_transform = target_trans.transform(y1_train)

    # define a Logistic Regression classifier
    model = LogisticRegression(solver='lbfgs', random_state=123, max_iter = 4000, multi_class = "ovr")

    # fit the Logistic Regression model
    model.fit(X=X_train_transform, y=y1_train_transform)

    print("Model Coefficient: ", model.coef_)
    print("Model Intercept: ", model.intercept_)
    print("Model Class: ", model.classes_)

    # server.model.coef_ = np.array(model.coef_)
    # server.model.intercept_ = model.intercept_
    # server.model.classes_ = np.array(list(model.classes_))

    # print(server.model.coef_)
    # print(server.model.intercept_)
    # print(server.model.classes_)

    projectId = "2HWWSM26k2fOYJGFAiAxCsF2TRF"
    projectSecret = "ab5fa9048e3166641c0a2726b6351230"
    endpoint = "https://ipfs.infura.io:5001"
    
    server_grads=list()
    server_grads.append(model.coef_.tolist())
    server_grads.append(model.intercept_.tolist())
    server_grads.append(model.classes_.tolist())
    print('Server Grads: ', server_grads)

    server_file = open("server.txt", "w+")

    # Saving the 2D array in a text file
    server_content = str(server_grads)

    server_file.write(server_content)
    server_file.close()
    file = {
        'server_file': 'server.txt',
    }

    ### ADD FILE TO IPFS AND SAVE THE HASH ###
    server_response = requests.post(endpoint + '/api/v0/add', files=file, auth=(projectId, projectSecret))
    # print("Here")
    print("Server Response:",server_response)
    server_hash = server_response.text.split(",")[1].split(":")[1].replace('"', '')
    print("Server Hash:", server_hash)
    return {
        "hash": server_hash
    }
# ********************************************************************************************************************


# Setting up the prediction route
@app.post("/train/")
async def train(input:Input):
#**************************FRAMINGHAM_DATASET******************************

    # projectId = "2HWWSM26k2fOYJGFAiAxCsF2TRF"
    # projectSecret = "ab5fa9048e3166641c0a2726b6351230"
    # endpoint = "https://ipfs.infura.io:5001"
    
    # print("Value of X: ", input.x)
    # print("s_hash: ", input.s_hash)

    # server_weight = []

    # for i in input.s_hash:
    #     # Prepare parameters for IPFS request
    #     params = {
    #         'arg': i
    #     }

    #     # Send a request to IPFS to retrieve data
    #     response2 = requests.post(endpoint + '/api/v0/cat', params=params, auth=(projectId, projectSecret))
    #     # print("Response2: ", response2)
    #     # print("Response2.text: ", response2.text)
        

    #     # Process the data from IPFS and create a list of clients
    #     with open(response2.text) as f:
    #         print(f"Value of f: {f}")
    #         for line in f:
    #             # print("Line: ", line)
    #             x = line.split(",")
    #             # print(x)
    #             # client=list()
    #             coef=list()
    #             intercept=list()
    #             classes=list()
    #             x[0]=x[0][2:]

    #             # Process and extract coefficients, intercept, and classes
    #             for i in range(13):
    #                 coef.append(float(x[i][1:]))
    #             x[13] = x[13][1:]
    #             x[13] = x[13][:-1]
    #             coef.append(float(x[13]))
    #             x[14]=x[14][2:]
    #             x[14]=x[14][:-1]
    #             intercept.append(float(x[14]))
    #             classes.append(int(x[15][-1]))
    #             classes.append(int(x[16][1]))
    #             server_weight.append(coef)
    #             server_weight.append(intercept)
    #             server_weight.append(classes)
    #             # print(server_weight)
                
    # print("server_weight: ", server_weight)
    # # server_weight[0] = np.array(server_weight[0])
    # # server_weight[1] = np.array(server_weight[1])
    # # server_weight[2] = list(server_weight[2])

    # # print("After: ", server_weight)
    # # print("server_weight: ", server_weight[0], server_weight[1], server_weight[2])
    # # server_weight[0] = np.array(server_weight[0])
    # # server_weight[1] = np.array(server_weight[1])
    # # server_weight[2] = np.array(server_weight[2])
    # # print("server_weight: ", server_weight[0], server_weight[1], server_weight[2])
    # # print("server.model: ", server.model.coef_, server.model.intercept_, server.model.classes_)
    # server.model.coef_ = np.array([server_weight[0]])
    # server.model.intercept_ = np.array(server_weight[1])
    # server.model.classes_ = np.array(list(set(server_weight[2])))
    # # print("server.model: ", server.model.coef_, server.model.intercept_, server.model.classes_)
    # client_list = list()
    # for i in range(input.x):                   # no of client bar loop cholbe
    #     temp = Client(i)
    #     temp.model.coef_ = server.model.coef_.copy()
    #     temp.model.intercept_ = server.model.intercept_.copy()
    #     temp.model.classes_ = server.model.classes_.copy()
    #     client_list.append(temp)          #  id diye client er object banaya push kortisis

    # df = pd.read_csv('model/framingham.csv')
    # # print(df.head())
    # df = df.dropna()
    # df.fillna(method='bfill', inplace=True)
    # df = shuffle(df)
    # y = df["TenYearCHD"]
    # X = df.drop(columns=['TenYearCHD', 'education'], axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # client_count = len(client_list)
    # # print(X_train[0:len(X_train)//client_count])
    # for i in range(len(client_list)):          #  proti ta client k dataset er kisu data diye train korse
    #     client_list[i].train(X_train[i*len(X_train)//client_count:(i+1)*len(X_train)//client_count], y_train[i*len(y_train)//client_count:(i+1)*len(y_train)//client_count])
    #     X_test_list.append(X_test[i*len(X_test)//client_count:(i+1)*len(X_test)//client_count])
    #     y_test_list.append(y_train[i*len(y_test)//client_count:(i+1)*len(y_test)//client_count])
    # # Assume client_list is a list containing models for all clients

    # client_hashes = {}

    # print(len(client_list))

    # for i, client in enumerate(client_list):
    #     client_info = list()
    #     client_info.append(client.model.coef_[0].tolist())
    #     client_info.append(client.model.intercept_.tolist())
    #     client_info.append(client.model.classes_.tolist())

    #     print('Array for client', i, ':\n', client_info)

    #     file_name = f"client{i}.txt"
    #     file = open(file_name, "w+")
    #     content = str(client_info)
    #     file.write(content)
    #     file.close()

    #     files = {f'file{i}': file_name}

    #     response = requests.post(endpoint + '/api/v0/add', files=files, auth=(projectId, projectSecret))
    #     print(response)
    #     client_hash = response.text.split(",")[1].split(":")[1].replace('"', '')
    #     print(client_hash)

    #     client_hashes[f"client{i}"] = client_hash

    # return {"data": client_hashes}

#**************************UNSW_NB15_DATASET******************************

    projectId = "2HWWSM26k2fOYJGFAiAxCsF2TRF"
    projectSecret = "ab5fa9048e3166641c0a2726b6351230"
    endpoint = "https://ipfs.infura.io:5001"
    
    print("Value of X: ", input.x)
    print("s_hash: ", input.s_hash)

    server_weight = []

    for i in input.s_hash:
        # Prepare parameters for IPFS request
        params = {
            'arg': i
        }

        # Send a request to IPFS to retrieve data
        response2 = requests.post(endpoint + '/api/v0/cat', params=params, auth=(projectId, projectSecret))
        # print("Response2: ", response2)
        # print("Response2.text: ", response2.text)
        

        # Process the data from IPFS and create a list of clients
        with open(response2.text) as f:
            print(f"Value of f: {f}")
            for line in f:
                print("Line: ", line)
                print("Type of Line: ", type(line))
                
                # Safely evaluate the string into a Python object
                data_list = ast.literal_eval(line)
                
                # Flatten and reshape the nested list
                coef = data_list[0]
                intercept = data_list[1]
                classes = data_list[2]

                # x = line.split(",")
                # print("X[0]: ", x[0])
                # # print(x)
                # # client=list()
                # coef=list()
                # intercept=list()
                # classes=list()
                # x[0]=x[0][2:]

                # # Process and extract coefficients, intercept, and classes
                # for i in range(13):
                #     coef.append(float(x[i][1:]))
                # x[13] = x[13][1:]
                # x[13] = x[13][:-1]
                # coef.append(float(x[13]))
                # x[14]=x[14][2:]
                # x[14]=x[14][:-1]
                # intercept.append(float(x[14]))
                # classes.append(int(x[15][-1]))
                # classes.append(int(x[16][1]))
                server_weight.append(coef)
                server_weight.append(intercept)
                server_weight.append(classes)
                # print(server_weight)
                
    print("server_weight: ", server_weight)
    # server_weight[0] = np.array(server_weight[0])
    # server_weight[1] = np.array(server_weight[1])
    # server_weight[2] = list(server_weight[2])

    # print("After: ", server_weight)
    # print("server_weight: ", server_weight[0], server_weight[1], server_weight[2])
    # server_weight[0] = np.array(server_weight[0])
    # server_weight[1] = np.array(server_weight[1])
    # server_weight[2] = np.array(server_weight[2])
    # print("server_weight: ", server_weight[0], server_weight[1], server_weight[2])
    # print("server.model: ", server.model.coef_, server.model.intercept_, server.model.classes_)
    server.model.coef_ = np.array(server_weight[0])
    server.model.intercept_ = np.array(server_weight[1])
    server.model.classes_ = np.array(list(set(server_weight[2])))
    # print("server.model: ", server.model.coef_, server.model.intercept_, server.model.classes_)
    print("Server Model Coefficient: ", server.model.coef_)
    print("Server Model Intercept: ", server.model.intercept_)
    print("Server Model Class: ", server.model.classes_)
    
    
    client_list = list()
    for i in range(input.x):                   # no of client bar loop cholbe
        temp = Client(i)
        temp.model.coef_ = server.model.coef_.copy()
        temp.model.intercept_ = server.model.intercept_.copy()
        temp.model.classes_ = server.model.classes_.copy()
        client_list.append(temp)          #  id diye client er object banaya push kortisis

    # df = pd.read_csv('model/framingham.csv')
    # # print(df.head())
    # df = df.dropna()
    # df.fillna(method='bfill', inplace=True)
    # df = shuffle(df)
    # y = df["TenYearCHD"]
    # X = df.drop(columns=['TenYearCHD', 'education'], axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)


    # Load data
    initial_data = pd.read_csv('model/UNSW_NB15_training-set.csv')
    test_data = pd.read_csv('model/UNSW_NB15_testing-set.csv')
    
    # Shuffle the rows
    initial_data = initial_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)

    # information of the data: 583 data points, 10 features' columns and 1 target column
    initial_data = initial_data.drop(axis=1, columns=['id'])
    initial_data = initial_data.drop(axis=1, columns=['proto'])
    initial_data = initial_data.drop(axis=1, columns=['service'])
    initial_data = initial_data.drop(axis=1, columns=['state'])
    
    test_data = test_data.drop(axis=1, columns=['id'])
    test_data = test_data.drop(axis=1, columns=['proto'])
    test_data = test_data.drop(axis=1, columns=['service'])
    test_data = test_data.drop(axis=1, columns=['state'])
    
    # initial_data.info()

    # Discard the rows with missing values
    data_to_use = initial_data.dropna()
    test_data = test_data.dropna()
    # data_to_use = data_to_use[47910: 48000]

    # Shape of the data: we could see that the number of rows remains the same as no null values were reported
    # print(data_to_use.shape)

    X = data_to_use.drop(axis=1, columns=['attack_cat']) # X is a dataframe
    X_train = X.drop(axis=1, columns=['label'])
    X_test = test_data.drop(axis=1, columns=['attack_cat']) # X_test is a dataframe
    X_test = X_test.drop(axis=1, columns=['label'])

    y1_train = data_to_use['attack_cat'].values # y is an array
    y2_train = data_to_use['label'].values
    y1_test = test_data['attack_cat'].values # y is an array
    y2_test = test_data['label'].values

    # determine categorical and numerical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns

    # define the transformation methods for the columns
    t = [('ohe', OneHotEncoder(drop='first'), categorical_cols),
        ('scale', StandardScaler(), numerical_cols)]

    col_trans = ColumnTransformer(transformers=t)

    # fit the transformation on training data
    col_trans.fit(X_train)

    X_train_transform = col_trans.transform(X_train)
    print("Shape of X_train_transform: ", X_train_transform.shape)
    X_test_transform = col_trans.transform(X_test)
    print("Shape of X_test_transform: ", X_test_transform.shape)

    # Define a LabelEncoder() transformation method and fit on y1_train
    target_trans = LabelEncoder()
    target_trans.fit(y1_train)

    # apply transformation method on y1_train and y1_test
    y1_train_transform = target_trans.transform(y1_train)
    y1_test_transform = target_trans.transform(y1_test)

    client_count = len(client_list)
    # print(X_train[0:len(X_train)//client_count])
    for i in range(len(client_list)):          #  proti ta client k dataset er kisu data diye train korse
        client_list[i].train(X_train_transform[i*len(X_train_transform)//client_count:(i+1)*len(X_train_transform)//client_count], y1_train_transform[i*len(y1_train_transform)//client_count:(i+1)*len(y1_train_transform)//client_count])
        X_test_list.append(X_test_transform[i*len(X_test_transform)//client_count:(i+1)*len(X_test_transform)//client_count])
        y_test_list.append(y1_test_transform[i*len(y1_test_transform)//client_count:(i+1)*len(y1_test_transform)//client_count])
    # Assume client_list is a list containing models for all clients

    client_hashes = {}

    print(len(client_list))

    for i, client in enumerate(client_list):
        client_info = list()
        client_info.append(client.model.coef_.tolist())
        client_info.append(client.model.intercept_.tolist())
        client_info.append(client.model.classes_.tolist())

        print('Array for client', i, ':\n', client_info)

        file_name = f"client{i}.txt"
        file = open(file_name, "w+")
        content = str(client_info)
        file.write(content)
        file.close()

        files = {f'file{i}': file_name}

        response = requests.post(endpoint + '/api/v0/add', files=files, auth=(projectId, projectSecret))
        print(response)
        client_hash = response.text.split(",")[1].split(":")[1].replace('"', '')
        print(client_hash)

        client_hashes[f"client{i}"] = client_hash

    return {"data": client_hashes}
# ************************************************************************************************************************


# Configuring the server host and port
if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')