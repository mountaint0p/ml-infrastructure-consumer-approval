import logging 
from abc import ABC, abstractmethod
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluating our models  
    """

    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the score of the model 
        """
        pass

class MSE(Evaluation):
    """Evaluation strategy for calculating mean squared error"""
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the mean squared error of the model 
        """
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e
            
class R2(Evaluation):
    """Evaluation strategy for calculating r2 score"""
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the r2 score of the model 
        """
        try:
            logging.info("Calculating R2")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2: {e}")
            raise e
            
class RMSE(Evaluation):
    """"Evaluation strategy for calculating root mean squared error"""
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the root mean squared error of the model 
        """
        try:
            logging.info("Calculating RMSE")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE: {e}")
            raise e
