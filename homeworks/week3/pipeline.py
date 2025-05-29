from prefect import flow, task
import pandas as pd
import mlflow



@task
def read_data(file_path):
    """
    Reads a parquet file and returns a DataFrame."""
    
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None
@task
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

@task
def train_model(data: pd.DataFrame):

    from sklearn.feature_extraction import DictVectorizer
    
    mlflow.set_experiment("nyc-taxi-experiment")
    
    data['duration'] = data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']
    data['duration'] = data['duration'].dt.total_seconds() / 60 
    data_dropped = data[(data['duration'] >= 1) & (data['duration'] <= 60)]

    # Convert the DataFrame to a list of dictionaries
    data_dicts = data_dropped[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    # Convert the IDs to strings
    for record in data_dicts:
        record['PULocationID'] = str(record['PULocationID'])
        record['DOLocationID'] = str(record['DOLocationID'])
    # Initialize the DictVectorizer
    dv = DictVectorizer()
    # Fit the DictVectorizer and transform the data
    X = dv.fit_transform(data_dicts)
    # Convert the sparse matrix to a dense format
    X_dense = X.toarray()
    # Convert the dense matrix to a DataFrame
    X_df = pd.DataFrame(X_dense, columns=dv.get_feature_names_out())


    from sklearn.linear_model import LinearRegression


    model = LinearRegression()
    model.fit(X, data_dropped['duration'])
    
    # Log the model with MLflow
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model")
        
        # model registiration
        mlflow.register_model("runs:/{}/model".format(mlflow.active_run().info.run_id), "nyc-taxi-duration-model")
    
    return model.intercept_, model.__sizeof__()


@flow
def data_pipeline(file_path):
    """
    A simple data pipeline that reads a parquet file and returns a DataFrame."""
    
   
    
    df = read_dataframe(file_path)
   
    if df is not None:
        print("Data read successfully.")
        intercept, sizeof = train_model(df)
        print(f"Model size: {sizeof}")
    else:
        print("Failed to read data.")
    
    return df



if __name__ == "__main__":
    file_path = "data/raw/yellow_tripdata_2023-03.parquet"  # Replace with your actual file path
    df = data_pipeline(file_path)
    
    if df is not None:
        print(df.shape)
    else:
        print("No data to display.")
