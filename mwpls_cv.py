from sklearn.metrics import mean_squared_error, max_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_predict, LeaveOneOut
from pandas import read_excel, DataFrame
from scipy.stats import shapiro
from scipy.signal import savgol_filter, detrend
from preproc_NIR import msc, snv, devise_bande, prep_log
from numpy import mean,sqrt,exp,log
import time
db=read_excel("orange-flowers-data-mean.xlsx")
Y=db['Y']
Y=[exp(i) for i in Y]
X=db.drop(['Unnamed: 0','Y'],axis=1)
X=msc(X.to_numpy())
X=DataFrame(savgol_filter(X,3,1,1))
X=DataFrame(detrend(X))
X.index=db['Unnamed: 0']
j=0
while True:
  x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=j)
  pls=PLSRegression(n_components=20,scale=True)
  pls.fit(x_train,y_train)
  ycv=cross_val_predict(PLSRegression(n_components=20,scale=False),x_train,y_train,cv=LeaveOneOut())
  if True:
    break
  j+=1
best_r2cv=r2_score(y_train, ycv)
bands=devise_bande(X,69)
while True:
  try:
      for i in bands:
        inp=X.drop(list(X.columns[i[0]:i[1]]),axis=1)
        j=0
        program_starts = time.time()
        while True:
          x_train, x_test, y_train, y_test = train_test_split(inp,Y,test_size=0.2,random_state=j)
          pls=PLSRegression(n_components=20,scale=True)
          pls.fit(x_train,y_train)
          ycv=cross_val_predict(PLSRegression(n_components=20,scale=False),x_train,y_train,cv=LeaveOneOut())
          now = time.time()
          run_time = now - program_starts
          if True:
            break
          if run_time>60:
              break
          j+=1
        if r2_score(y_train, ycv)>=best_r2cv:
          best_r2cv=r2_score(y_train,pls.predict(x_train))
          best_i=i
      X=X.drop(list(X.columns[best_i[0]:best_i[1]]),axis=1)
      bands.remove(best_i)
  except: # ValueError
      break
print("Selected wl : ",len(X.columns))
print("R² train : ",r2_score(y_train, ycv))