import pickle
import pandas as pd
from datetime import date
import sys

year = str(sys.argv[1]) # '2021'
month = str(sys.argv[2]) # 03

print("Year : " + year)
print("Month : " + month)

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):

    year = str(date.today().year)
    month = str(date.today().year)

    print(filename)
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    df['ride_id'] = year + month + df.index.astype('str')
    
    return df

df = read_data('https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_' + str(year) + '-' + str(month) + '.parquet')

print(df.duration.describe())

dicts = df[categorical].to_dict(orient = 'records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)

output_file = "file.parquet"

df_result = df[['ride_id', 'duration']]

df_result.to_parquet(
    output_file,
    engine = 'pyarrow',
    compression = None,
    index = False
)



