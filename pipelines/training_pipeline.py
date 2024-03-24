from zenml import pipeline
from steps.clean_data import clean_df
from steps.ingest_data import ingest_data
from steps.evaluation import evalute_model
from steps.model_train import train_model


@pipeline(enable_cache=False)
def training_pipeline(data_path: str):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2, rmse = evalute_model(model, X_test, y_test)