import logging 
from typing import Tuple

import pandas as pd 
from zenml import step 
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml.client import Client
import mlflow

from src.evaluation import MSE, R2, RMSE

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evalute_model(model: RegressorMixin,
        X_test: pd.DataFrame, 
        y_test: pd.DataFrame
    ) -> Tuple[
        Annotated[float, "r2"], 
        Annotated[float, "rmse"]
    ]:
    """
    Evaluates the model on the ingested data 
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("MSE", mse)

        r2_class = R2()
        r2 = r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric("R2", r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("RMSE", rmse)

        return r2, rmse
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e