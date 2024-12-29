import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from  sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
import joblib


class Regressor:
    def __init__(self):

        return
            
    def regression(self, *args, **kwargs):
        raise NotImplementedError("Die Regression-Methode muss in der Unterklasse implementiert werden.")


    def __call__(self, *args, **kwargs):

        return self.regression(*args,**kwargs)
    


class GaussianRegressor(Regressor):
    def __init__(self, kernel,alpha):
        super().__init__()

        self.kernel= kernel
        self.alpha= alpha

        return
    

    def Gaussian_plot_visualization(self,X,y,X_train,y_train,gpr):
        gpr_loaded = joblib.load(gpr)
        y_predictions,y_variance= gpr_loaded.predict(X, return_std=True)

        y_predictions = y_predictions.flatten()
        y_train = y_train.reshape(-1,)

        y_train= y_train.reshape(-1,1)
        X= X.flatten()

        #plt.plot(X, y, label="Ground Truth", linestyle="dotted")
        plt.scatter(X_train, y_train, label="Observations")
        plt.scatter(X, y_predictions, color= 'b',label="Mean prediction")
        sorted_indices = np.argsort(X)  # Indizes zum Sortieren von X

        X=X[sorted_indices]
        y_predictions=y_predictions[sorted_indices]
        y_variance=y_variance[sorted_indices]

        print(y_predictions)
        print(y_variance)
        plt.fill_between(X.ravel(), y_predictions - (1e7 * y_variance), y_predictions + (1e7 * y_variance), color='C1', alpha= 0.3, interpolate=True, label= r"95% confidence interval")
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Gaussian process regression on noise-free dataset")
        plt.show()


        return
        
    def regression(self,X,y):

    
        gpr= GaussianProcessRegressor(kernel=self.kernel,
        alpha= self.alpha).fit(X,y)
        r2= gpr.score(X,y)
        model= joblib.dump(gpr, 'gpr_model.pkl')

        return r2


class MyRegression(Regressor):
    def __init__(self):
        super().__init__()

        self.Regressor= GaussianRegressor( alpha=1e-10, kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
)

        return
    
    def regression(self,X,y):

        r2= self.Regressor(X,y)
        
        return r2
    
    def plot(self, X,y,X_train,y_train, gpr):

        self.Regressor.Gaussian_plot_visualization(X,y,X_train,y_train,gpr)

        return
    

if __name__== '__main__':
    

    diabetes = load_diabetes()
    y= diabetes.target[:]
    X= diabetes.data[:,:]
    y= y.reshape(-1,1)

    df_diabetes= pd.DataFrame(np.hstack((X,y)))

    df_X= df_diabetes.iloc[:,:-1]
    df_y= df_diabetes.iloc[:,-1]


    pca= PCA(n_components=1,copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', n_oversamples=10, power_iteration_normalizer='auto', random_state=None)
    reduced_X= pd.DataFrame(pca.fit_transform(df_X))

    print(reduced_X.shape)
    print(type(reduced_X))

    df_Xtrain, df_Xtest, df_ytrain, df_ytest= train_test_split(reduced_X,df_y,test_size=0.2,random_state=None, shuffle=True, stratify=None)

    X=np.array(reduced_X)
    y=np.array(df_y)
    X_train= np.array(df_Xtrain)
    X_test= np.array(df_Xtest)
    y_train= np.array(df_ytrain)
    y_test= np.array(df_ytest)




    print(df_diabetes)
    first_model= MyRegression()
    model= first_model(df_Xtrain, df_ytrain)

    print(f' the r2 score on the train set is: {model}')

    pickel= 'C:\\Users\\badar\\OneDrive\Dokumente\\Reproduzieren\\HIWI\\gpr_model.pkl'
    first_model.plot(X,y,X_train,y_train, pickel)




    