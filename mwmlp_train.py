from sklearn.metrics import mean_squared_error, max_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from statsmodels.multivariate.pca import PCA
from sklearn.model_selection import train_test_split
from pandas import read_excel, DataFrame
from scipy.stats import shapiro
from scipy.signal import savgol_filter
from preproc_NIR import msc, snv, devise_bande
from numpy import mean,sqrt
import time
db=read_excel("orange-folwers-data.xlsx")
Y=db['Y']
#Y=[sqrt(i) for i in Y]
X=db.drop(['Unnamed: 0','Y'],axis=1)
X=msc(savgol_filter(X,3,1,1))
X.index=db['Unnamed: 0']
j=0
while True:
  pc=PCA(X,X.shape[0],method='nipals')
  x_train, x_test, y_train, y_test = train_test_split(DataFrame(pc.factors),Y,test_size=0.2,random_state=j)
  nnm=MLPRegressor(hidden_layer_sizes=(100,),solver='adam',activation='identity',alpha=0.01,learning_rate='adaptive')
  nnm.fit(x_train,y_train)
  if r2_score(y_train,nnm.predict(x_train))>0:
    break
  j+=1
best_r2c=r2_score(y_train,nnm.predict(x_train))
bands=devise_bande(X,69)
while True:
  try:
      for i in bands:
        inp=X.drop(list(X.columns[i[0]:i[1]]),axis=1)
        pc=PCA(inp,inp.shape[0],method='nipals')
        j=0
        program_starts = time.time()
        while True:
          x_train, x_test, y_train, y_test = train_test_split(DataFrame(pc.factors),Y,test_size=0.2,random_state=j)
          nnm=MLPRegressor(hidden_layer_sizes=(100,),solver='sgd',activation='relu',alpha=0.01,learning_rate='adaptive')
          nnm.fit(x_train,y_train)
          now = time.time()
          run_time = now - program_starts
          if r2_score(y_train,nnm.predict(x_train))>0:
            break
          if run_time>60:
              break
          j+=1
        if r2_score(y_train,nnm.predict(x_train))>=best_r2c:
          best_r2c=r2_score(y_train,nnm.predict(x_train))
          best_i=i
      X=X.drop(list(X.columns[best_i[0]:best_i[1]]),axis=1)
      bands.remove(best_i)
  except: # ValueError
      break
print("Selected wl : ",len(X.columns))
print("R² train : ",r2_score(y_train,nnm.predict(x_train)))