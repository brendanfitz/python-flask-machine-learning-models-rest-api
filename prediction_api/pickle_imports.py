import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from prediction_api.ml_models import luther_util
from sklearn.ensemble import RandomForestClassifier
import time
import boto3
import warnings
from statsmodels.regression.linear_model import OLSResults
from prediction_api.ml_models.db import ml_db
warnings.simplefilter("ignore", UserWarning)

LOCAL_DIRECTORY = 'prediction_api/static/pickles/'
AWS_ACCESS_KEY_ID = os.environ.get('prediction_api_AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.environ.get('prediction_api_AWS_SECRET_KEY')


if not os.path.exists(LOCAL_DIRECTORY):
    os.mkdir(LOCAL_DIRECTORY)

import sys
sys.path.append('prediction_api/ml_models')

def aws_download(object_name, filename=None,
    bucket_name='metis-projects',
    bucket_directory='pickles',
    local_directory='prediction_api/static/pickles'):

    if bucket_directory:
        object_path = bucket_directory + '/' + object_name
    else:
        object_path = object_name

    if filename:
        download_path =  local_directory + '/' + filename
    else:
        download_path =  local_directory + '/' + object_name

    if not os.path.isfile(download_path):
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_KEY,
        )
        s3.download_file(bucket_name, object_path, download_path)
    return download_path


class Pickle_Imports:

    def __init__(self):
        self.kickstarter_vectorizer = None
        self.kickstarter_model = None
        self.titantic_model = None
        self.nhl_goals_model = None

    def titantic_downloads(self):
        start = time.time()
        filename = 'titanic_model.pkl'
        with open(aws_download(filename), 'rb') as f:
            self.titantic_model = pickle.load(f)
        end = time.time()
        print('Titantic Model: {:,.4f} seconds'.format(end - start))

    def nhl_downloads(self):
        start = time.time()
        filename = 'nhl_goals_regression_model.pkl'
        with open(aws_download(filename), 'rb') as f:
            self.nhl_goals_model = OLSResults.load(f)
        end = time.time()
        print('NHL Goals Model: {:,.4f} seconds'.format(end - start))

    def pickle_isdownloaded(self, filename):
        filepath = os.path.join(LOCAL_DIRECTORY, filename)
        return os.path.isfile(filepath)

    def all_models_pickles_are_downloaded(self, model):
        model_data = next(filter(lambda x: model == x['id'], ml_db))
        pkl_list = model_data['pickles']
        return all([self.pickle_isdownloaded(filename) for filename in pkl_list])
