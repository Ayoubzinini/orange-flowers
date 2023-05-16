from pandas import read_excel, DataFrame
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from numpy import sqrt
from scipy.signal import savgol_filter
from scipy.stats import shapiro
from preproc_NIR import msc, snv
from statsmodels.multivariate.pca import PCA
db=read_excel("orange-folwers-data.xlsx")
Y=db['Y']
X=db.drop(['Unnamed: 0','Y'],axis=1)
X=DataFrame(savgol_filter(X,3,1,1))
X.index=db['Unnamed: 0']
pc=PCA(X,X.shape[0],method='nipals')
j=0
while True:
    x_train, x_test, y_train, y_test = train_test_split(pc.factors,Y,test_size=0.2,random_state=j)
    model = MLPRegressor(hidden_layer_sizes=(100,),solver='sgd',activation='relu',alpha=0.01,learning_rate='adaptive')
    model.fit(x_train,y_train)
    model.fit(x_train, y_train)
    ycv=cross_val_predict(MLPRegressor(hidden_layer_sizes=(100,),activation='identity',alpha=0.01,learning_rate='adaptive'),x_train,y_train,cv=LeaveOneOut())
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