import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

import datetime
from datetime import date
import dateutil.relativedelta

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train = True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient = 'records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared = False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def get_paths(date_input):

    if date_input is None:
        date_input = date.today()

    d = datetime.datetime.strptime(str(date_input), "%Y-%m-%d")
    f_training = d - dateutil.relativedelta.relativedelta(months = 2)
    f_validation = d - dateutil.relativedelta.relativedelta(months = 1)

    f_training = f_training.strftime("%Y-%m")
    f_validation = f_validation.strftime("%Y-%m")

    #concatenate path
    path_dataset = '../data/fhv_tripdata_{0}.parquet'

    return path_dataset.format(f_training), path_dataset.format(f_validation)

#train_path, val_path = get_paths(date).result()

def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient = 'records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared = False)
    print(f"The MSE of validation is: {mse}")
    return

@flow(task_runner = SequentialTaskRunner())
def main(date_input = None):

    categorical = ['PUlocationID', 'DOlocationID']

    train_path, val_path = get_paths(date_input).result()

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical).result()

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False).result()

    # train the model
    lr, dv = train_model(df_train_processed, categorical)
    run_model(df_val_processed, categorical, dv, lr)

    #sorted(model_list, reverse=False)[0]

main(date_input = "2021-08-15")

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule,CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

#from prefect.orion.schemas.schedules import CronSchedule

from datetime import timedelta

DeploymentSpec(
    flow = main,
    name = "home_homework_prefect",
    schedule = CronSchedule(cron = "0 9 15 * *"),
    flow_runner = SubprocessFlowRunner(), #local
    tags = ["ml"]
)