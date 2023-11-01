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
from sklearn.preprocessing import StandardScaler

server = Server()
X_test_list = []
y_test_list = []

class Input(BaseModel):
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
    projectId = "2HWWSM26k2fOYJGFAiAxCsF2TRF"
    projectSecret = "ab5fa9048e3166641c0a2726b6351230"
    endpoint = "https://ipfs.infura.io:5001"
    # print(await request.json())

    
    res = await request.json()      # Retrieve JSON data from the HTTP request
    print("result")
    print(res)

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
                x = line.split(",")
                # print(x)
                client=list()
                coef=list()
                intercept=list()
                classes=list()
                x[0]=x[0][2:]

                # Process and extract coefficients, intercept, and classes
                for i in range(13):
                    coef.append(float(x[i][1:]))
                x[13] = x[13][1:]
                x[13] = x[13][:-1]
                coef.append(float(x[13]))
                x[14]=x[14][2:]
                x[14]=x[14][:-1]
                intercept.append(float(x[14]))
                classes.append(int(x[15][-1]))
                classes.append(int(x[16][1]))
                client.append(coef)
                client.append(intercept)
                client.append(classes)
                print(client)
                clients.append(client)

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
    server_grads.append(server.model.coef_[0].tolist())
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
    model = LogisticRegression()
    df = pd.read_csv('model/framingham.csv')
    df = df.dropna()
    df.fillna(method='bfill', inplace=True)
    data = df[28:30]

    y = data["TenYearCHD"]
    X = data.drop(columns=['TenYearCHD', 'education'], axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)

    model.fit(X_train, y)

    print(model.coef_)
    print(model.intercept_)
    print(model.classes_)

    server.model.coef_ = np.array(model.coef_)
    server.model.intercept_ = model.intercept_
    server.model.classes_ = np.array(list(model.classes_))

    print(server.model.coef_)
    print(server.model.intercept_)
    print(server.model.classes_)

    projectId = "2HWWSM26k2fOYJGFAiAxCsF2TRF"
    projectSecret = "ab5fa9048e3166641c0a2726b6351230"
    endpoint = "https://ipfs.infura.io:5001"
    
    server_grads=list()
    server_grads.append(server.model.coef_[0].tolist())
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
        "hash": server_hash
    }

# Setting up the prediction route
@app.post("/train/")
async def train(input:Input):
    projectId = "2HWWSM26k2fOYJGFAiAxCsF2TRF"
    projectSecret = "ab5fa9048e3166641c0a2726b6351230"
    endpoint = "https://ipfs.infura.io:5001"
    
    print(input.x)

    client_list = list()
    for i in range(input.x):                   # no of client bar loop cholbe
        temp = Client(i)
        temp.model.coef_ = server.model.coef_.copy()
        temp.model.intercept_ = server.model.intercept_.copy()
        temp.model.classes_ = server.model.classes_.copy()
        client_list.append(temp)          #  id diye client er object banaya push kortisis

    df = pd.read_csv('model/framingham.csv')
    # print(df.head())
    df = df.dropna()
    df.fillna(method='bfill', inplace=True)
    df = shuffle(df)
    y = df["TenYearCHD"]
    X = df.drop(columns=['TenYearCHD', 'education'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    client_count = len(client_list)
    print(X_train[0:len(X_train)//client_count])
    for i in range(len(client_list)):          #  proti ta client k dataset er kisu data diye train korse
        client_list[i].train(X_train[i*len(X_train)//client_count:(i+1)*len(X_train)//client_count], y_train[i*len(y_train)//client_count:(i+1)*len(y_train)//client_count])
        X_test_list.append(X_test[i*len(X_test)//client_count:(i+1)*len(X_test)//client_count])
        y_test_list.append(y_train[i*len(y_test)//client_count:(i+1)*len(y_test)//client_count])
    # Assume client_list is a list containing models for all clients

    client_hashes = {}

    print(len(client_list))

    for i, client in enumerate(client_list):
        client_info = list()
        client_info.append(client.model.coef_[0].tolist())
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


# Configuring the server host and port
if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')