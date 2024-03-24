import logging 
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class defining model 
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model 
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear regression model 
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the linear regression model 
        """

        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg 
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e