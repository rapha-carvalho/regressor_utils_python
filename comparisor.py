import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

from .metrics import ModelMetrics


class RegressorModel():
    '''
    This class has functions to select, fit and evaluate all regression models.
    '''
    def __init__(self, train_features, train_target, test_features, test_target):
        self.train_features = train_features
        self.train_target = train_target
        self.test_features = test_features
        self.test_target = test_target
        #Instantiate the ModelMetrics class (class that evaluates a model's performance)
        self.metric = ModelMetrics(self.train_features, self.train_target, self.test_features, self.test_target)


    def linear_regression(self, params: dict = None):
        '''
        Instantiate the LinearRegression model, fits the model and return model's performance.
        '''
        if params == None:
            lr = LinearRegression()
        else:
            lr = LinearRegression(**params)
        lr.fit(self.train_features, self.train_target)
        self.metric.set_model(lr)
        self.metric.all_metrics()
        errors = self.metric.get_errors()
        errors['ESTIMATOR'] = 'LinearRegression'
        lr_df = pd.DataFrame.from_dict(errors)
        return lr_df

    def ridge_regression(self, params: dict = None):
        '''
        Instantiate the Ridge model, fits the model and return model's performance.
        '''
        if params == None:
            rdg = Ridge()
        else:
            rdg = Ridge(**params)
        rdg.fit(self.train_features, self.train_target)
        self.metric.set_model(rdg)
        self.metric.all_metrics()
        errors = self.metric.get_errors()
        errors['ESTIMATOR'] = 'Ridge'
        rdg_df = pd.DataFrame.from_dict(errors)
        return rdg_df

    def lasso_regression(self, params: dict = None):
        '''
        Instantiate the Lasso model, fits the model and return model's performance.
        '''
        if params == None:
            lasso = Lasso()
        else:
            lasso = Lasso(**params)
        lasso.fit(self.train_features, self.train_target)
        self.metric.set_model(lasso)
        self.metric.all_metrics()
        errors = self.metric.get_errors()
        errors['ESTIMATOR'] = 'Lasso'
        lasso_df = pd.DataFrame.from_dict(errors)
        return lasso_df

    def decisiontree_regression(self, params: dict = None):
        '''
        Instantiate the DecisionTreeRegressor model, fits the model and return model's performance.
        '''
        if params == None:
            dt = DecisionTreeRegressor()
        else:
            dt = DecisionTreeRegressor(**params)
        dt.fit(self.train_features, self.train_target)
        self.metric.set_model(dt)
        self.metric.all_metrics()
        errors = self.metric.get_errors()
        errors['ESTIMATOR'] = 'DecisionTreeRegressor'
        dt_df = pd.DataFrame.from_dict(errors)
        return dt_df

    def linear_svr_regression(self, params: dict = None):
        '''
        Instantiate the LinearSVR model, fits the model and return model's performance.
        '''
        if params == None:
            linear_svr = LinearSVR()
        else:
            linear_svr = LinearSVR(**params)
        linear_svr.fit(self.train_features, self.train_target)
        self.metric.set_model(linear_svr)
        self.metric.all_metrics()
        errors = self.metric.get_errors()
        errors['ESTIMATOR'] = 'LinearSVR'
        linear_svr_df = pd.DataFrame.from_dict(errors)
        return linear_svr_df

    def random_forest_regressor(self, params: dict = None):
        '''
        Instantiate the RandomForestRegressor model, fits the model and return model's performance.
        '''
        if params == None:
            rf = RandomForestRegressor()
        else:
            rf = RandomForestRegressor(**params)
        rf.fit(self.train_features, self.train_target)
        self.metric.set_model(rf)
        self.metric.all_metrics()
        errors = self.metric.get_errors()
        errors['ESTIMATOR'] = 'RandomForestRegressor'
        rf_df = pd.DataFrame.from_dict(errors)
        return rf_df

    def gradient_boosting_regressor(self, params: dict = None):
        '''
        Instantiate the GradientBoostingRegressor model, fits the model and return model's performance.
        '''
        if params == None:
            gbr = GradientBoostingRegressor()
        else:
            gbr = GradientBoostingRegressor(**params)
        gbr.fit(self.train_features, self.train_target)
        self.metric.set_model(gbr)
        self.metric.all_metrics()
        errors = self.metric.get_errors()
        errors['ESTIMATOR'] = 'GradientBoostingRegressor'
        gbr_df = pd.DataFrame.from_dict(errors)
        return gbr_df

    def mlp_regression(self, params: dict = None):
        '''
        Instantiate the MLPRegressor model, fits the model and return model's performance.
        '''
        if params == None:
            mlp = MLPRegressor()
        else:
            mlp = MLPRegressor(**params)
        mlp.fit(self.train_features, self.train_target)
        self.metric.set_model(mlp)
        self.metric.all_metrics()
        errors = self.metric.get_errors()
        errors['ESTIMATOR'] = 'MLPRegressor'
        mlp_df = pd.DataFrame.from_dict(errors)
        return mlp_df


    def summary(self):
        '''
        Creates a dataframe containing all the performance results of the models that were evaluated.
        '''
        lr = self.linear_regression()
        rdg = self.ridge_regression()
        lasso = self.lasso_regression()
        rf = self.random_forest_regressor()
        dt = self.decisiontree_regression()
        linear_svr = self.linear_svr_regression()
        gbr = self.gradient_boosting_regressor()
        mlp = self.mlp_regression()
        join = pd.concat([lr, rdg, lasso, dt, linear_svr, rf, gbr, mlp])
        table = join.set_index('ESTIMATOR')
        return table.to_string()
