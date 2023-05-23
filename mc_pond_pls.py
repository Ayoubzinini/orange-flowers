from sklearn.metrics import mean_squared_error, max_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from pandas import read_excel, DataFrame
from scipy.stats import shapiro, f_oneway
from scipy.signal import savgol_filter
from preproc_NIR import msc, snv, devise_bande, prep_log
from missed_metrics import vip
from numpy import mean,sqrt,exp,log
import time
db=read_excel("orange-folwers-data.xlsx")
Y=db['Y']
Y=[sqrt(i) for i in Y]
X=db.drop(['Unnamed: 0','Y'],axis=1)
wl=X.columns
#X=prep_log(X)
X=DataFrame(savgol_filter(msc(X.to_numpy()),3,1,1))
X.index=db['Unnamed: 0']
selected=[]
for wave in range(len(wl)):
    j=0
    while True:
      x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=j)
      pls=PLSRegression(n_components=4,scale=False)
      pls.fit(x_train,y_train)
      dr2c=r2_score(y_train,pls.predict(x_train))
      income_groups=[y_train,y_test]
      s,p=f_oneway(*income_groups)
      if p<0.05 and dr2c>0:
        break
      j+=1
    X[X.columns[0]]=X[X.columns[0]]*2
    while True:
      x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=j)
      pls=PLSRegression(n_components=4,scale=False)
      pls.fit(x_train,y_train)
      fir2c=r2_score(y_train,pls.predict(x_train))
      fir2c=r2_score(y_test,pls.predict(x_test))
      fir2c=r2_score(y_train,cross_val_predict(PLSRegression(n_components=4,scale=False),x_train,y_train,cv=LeaveOneOut()))
      income_groups=[y_train,y_test]
      s,p=f_oneway(*income_groups)
      if p<0.05 and fir2c>0:
        break
      j+=1
    if fir2c>dr2c:
        kr2c=fir2c
    else:
        X[X.columns[0]]=X[X.columns[0]]/2
        kr2c=dr2c
    for i in X.columns[1:len(X.columns)]:
        X[i]=X[i]*2
        j=0
        while True:
          x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=j)
          pls=PLSRegression(n_components=4,scale=False)
          pls.fit(x_train,y_train)
          nir2c=r2_score(y_train,pls.predict(x_train))
          income_groups=[y_train,y_test]
          s,p=f_oneway(*income_groups)
          if p<0.05 and nir2c>0:
            break
          j+=1
        if nir2c>kr2c:
            kr2c=nir2c
            bi=i
            bpls=pls
            bxc,bxt,byc,byt=x_train,x_test,y_train,y_test
            kvip=vip(x_test, y_test, pls)
        else:
            X[i]=X[i]/2
    selected.append(bi)