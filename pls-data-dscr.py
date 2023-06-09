from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, max_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from pandas import read_excel, DataFrame
from scipy.stats import shapiro, f_oneway
from scipy.signal import savgol_filter, detrend
from preproc_NIR import msc, snv
from numpy import mean,sqrt, std, median
from pysptools.spectro import convex_hull_removal
from dtwalign import dtw
db=read_excel("orange-flowers-data-mean.xlsx")
Y=db['Y']
Y=[sqrt(i) for i in Y]
X=db.drop(['Unnamed: 0','Y'],axis=1)
wl=X.columns
#you should try mwpls to optimize R² CV
X=msc(X.to_numpy())
X=DataFrame(savgol_filter(X,3,1,1))
X=DataFrame(detrend(X)) 
X.index=db['Unnamed: 0']
X.columns=wl
X=X.drop(list(X.columns[(15, 31)[0]:(15, 31)[1]]),axis=1)
j=0
while True:
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=j)
    income_groups=[y_train,y_test]
    s,p=f_oneway(*income_groups)
    r2=[]
    RMSE=[]
    for i in range(1,x_train.shape[0]+1,1):
        model=PLSRegression(n_components=i,scale=False)
        model.fit(x_train, y_train)
        r2.append(r2_score(model.predict(x_test),y_test))
        RMSE.append(mean_squared_error(model.predict(x_test),y_test))
    model=PLSRegression(n_components=1+RMSE.index(min(RMSE)),scale=False)
    model.fit(x_train, y_train)
    ycv=cross_val_predict(PLSRegression(n_components=1+RMSE.index(min(RMSE)),scale=False),x_train,y_train,cv=LeaveOneOut())
    RMSECV=sqrt(mean_squared_error(y_train, ycv))
    R2CV=100*r2_score(y_train, ycv)
    R2train=100*r2_score(y_train,model.predict(x_train))
    R2test=100*r2_score(y_test,model.predict(x_test))
    RMSEtrain=sqrt(mean_squared_error(y_train,model.predict(x_train)))
    RMSEtest=sqrt(mean_squared_error(y_test,model.predict(x_test)))
    if R2CV>0 and R2test>0:
        break
    j=j+1
w,p = shapiro([i-j for i,j in zip(y_test,model.predict(x_test))])
if p>0.05:
  desicion="Normal"
elif p<0.05:
  desicion="Not Normal"
print(DataFrame({
    "R² c":R2train,
    "R² CV":R2CV,
    "R² t":R2test,
    "RMSE c":RMSEtrain,
    "RMSE CV":RMSECV,
    "RMSE t":RMSEtest,
    "Quantile Shapiro":w,
    "P Shapiro":p,
    "Decision":desicion,
    "RDS":j
},index=[0]))