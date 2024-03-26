# FastML

## Overview

The idea behind this project is to build a web application that utilizes the power of [microsoft/FLAML](https://github.com/microsoft/FLAML) library which is built for Automated Machine Learning tasks. I have found this library to be very useful for building and deploying Machine Learning projects at a rapid pace. The FLAML library is a low-code library with many integrations to other libraries like pandas, numpy and scikit-learn which makes data preparation simpler. FLAML mainly uses Tree based learning models like XGBoost, LightGBM, Catboost and Random Forest for training also tree based models don't require intensive data preprocessing steps and can also working with missing data. 

My goal is to build an application that enables users to build their own machine learning models using their own data more privately, securely and most importantly without needed hardware that costs a lot. This application can run on any 4-core laptop from the last 4 years without any any slow-down.

## User Flow

### 1. Generate Token
- **Objective:** Generate a unique token to be used as a job ID for subsequent interactions.
- User sends a GET request to the `/generate-token` endpoint.
- The server generates a unique token and returns it to the user.
- The user stores this token for future use.

### 2. Upload Dataset
- **Objective:** Upload a dataset in CSV format using the generated token.
- User sends a POST request to the `/upload-data/{token}` endpoint, attaching the CSV file.
- User includes the generated token in the URL path.
- The server receives the CSV file, processes it, and associates it with the provided token.

### 3. Start Training
- **Objective:** Initiate the training process by providing additional training parameters and target feature name.
- User sends a POST request to the `/start-training/{token}` endpoint, including the token in the URL path.
- User provides training parameters such as algorithm type, hyperparameters, and other relevant details.
- User specifies the target feature name.
- The server starts the training process using the uploaded dataset and provided parameters.

### 4. Check Job Status
- **Objective:** Monitor the status of the whole job process.
- User sends a GET request to the `/check-status/{token}` endpoint, including the token in the URL path.
- The server will send the status along with other metadata like `job-token` and `timestamp` of last status update.

### 5. View Evaluation Results
- **Objective:** Access evaluation results after the training process is complete.
- User sends a GET request to the `/evaluate-training/{token}` endpoint, including the token in the URL path.
- The server retrieves evaluation metrics and results from the trained model.
- The server responds with the evaluation results, including metrics like accuracy, precision, recall, etc.

