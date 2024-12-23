import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


class Regressor:
    def __init__(self):

        self.regressors = {}

        return

    def add_module(self, name, regressors):
            if not isinstance(regressors, Regressor) and regressors is not None:
                raise TypeError(f"{regressors} ist kein gültiges Modul.")
            self.regressors[name] = regressors

            
    def regression(self, *args, **kwargs):
        raise NotImplementedError("Die Regression-Methode muss in der Unterklasse implementiert werden.")


    def __call__(self, *args, **kwds):

        return self.regression()
    


class GaussianRegressor(Regressor):
    def __init__(self):
        super().__init__()

        self.kernel= None
        self.alpha= None

        return
    
    def regression(self,X,y):

        self.kernel= WhiteKernel()
        self.alpha=1e-10
        gpr = GaussianProcessRegressor(kernel=self.kernel,
        alpha=self.alpha).fit(X, y)
        metric_value = gpr.score(X,y)

        return metric_value
    


class MyRegression(Regressor):
    def __init__(self):
        super().__init__()

        self.regressor= GaussianRegressor()
        self.add_module('regressor', self.regressor)
        return
    
    def regression(self,X,y):

        Score= self.regressor(X,y)
        
        return Score
    

if __name__== '__main__':
    
    X= np.random.rand(20,1)
    y=np.random.normal(0,1,size=(20,1))
    first_model= MyRegression()
    score=first_model(X,y)

