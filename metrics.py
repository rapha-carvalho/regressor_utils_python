import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelMetrics():
    '''
    This class has functions to calculate a model's accuracy.
    '''
    def __init__(self, train_features, train_target, test_features, test_target, model):
        self.model = model
        self.train_features = train_features
        self.train_target = train_target
        self.test_features = test_features
        self.test_target = test_target

    def calc_ISE(self):
        '''
        Returns the in-sample R² and RMSE;
        Assumes model is already fitted.
        '''
        predictions = self.model.predict(self.train_features)
        mse = mean_squared_error(self.train_target, predictions)
        rmse = np.sqrt(mse)
        return model.score(self.train_features, self.train_target), rmse

    def calc_OSE(self):
        '''
        Returns the out-of-sample R² and RMSE;
        Assumes model is already fitted.
        '''
        predictions = self.model.predict(self.test_features)
        mse = mean_squared_error(self.test_target, predictions)
        rmse = np.sqrt(mse)
        return model.score(self.test_features, self.test_target), rmse

    def calc_train_error(self):
        '''
        Returns in-sample error for already fitted model.
        '''
        predictions = self.model.predict(self.train_features)
        mse = mean_squared_error(self.train_target, predictions)
        rmse = np.sqrt(mse)
        return rmse

    def calc_validation_error(self):
        '''
        Returns out-of-sample error for already fitted model.
        '''
        predictions = self.model.predict(self.test_features)
        mse = mean_squared_error(self.test_target, predictions)
        rmse = np.sqrt(mse)
        return rmse

    def model_error(self):
        '''
        Returns the RMSE for in-sample error and out-of-sample calc_train_error on an already fitted model.
        '''
        train_error = self.calc_train_error()
        validation_error = self.calc_validation_error()
        return train_error, validation_error

    def calc_MAE(self):
        '''
        Returns the Mean Absolute Error of an already fitted model.
        '''
        predictions = self.model.predict(self.test_features)
        mae = mean_absolute_error(self.test_target, predictions)
        return mae

    def calc_MAPE(self):
        '''
        Returns the Mean Absolute Percentual Error of an already fitted model.
        '''
        predictions = self.model.predict(self.test_features)
        errors = abs(predictions - self.test_target)
        mape = (errors / self.test_target) * 100
        return np.mean(mape)

    def calc_accuracy(self):
        '''
        Returns the accuracy (%) of an already fitted model.
        '''
        predictions = self.model.predict(self.test_features)
        mape = self.calc_MAPE()
        accuracy = 100 - mape
        return str(accuracy) + '%'
