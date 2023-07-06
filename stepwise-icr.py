from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_predict, LeaveOneOut
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import RFECV
from pandas import DataFrame, read_excel, concat
from scipy.stats import shapiro, f_oneway
from scipy.signal import savgol_filter, detrend
from preproc_NIR import simple_moving_average, snv, msc, osc
db=read_excel("orange-flowers-data-mean.xlsx")
Y=db['Y']
X=db.drop(['Unnamed: 0','Y'],axis=1)
wl=X.columns
#"""
X=msc(X.to_numpy())
X=DataFrame(savgol_filter(X,3,1,1))
#"""
X=DataFrame(detrend(X))
X.index=db['Unnamed: 0']
X.columns=wl
ica=FastICA(n_components=X.shape[0])
ica.fit_transform(X)
pc=DataFrame(ica.components_)
j=0
while True:
    x_train, x_test, y_train, y_test = train_test_split(pc,Y,test_size=0.2,random_state=j)
    income_groups=[y_train,y_test]
    s,p=f_oneway(*income_groups)
    if p<0.05:
        break
    j+=1
estm=RFECV(LinearRegression(),cv=LeaveOneOut())
estm.fit(x_train,y_train)
print(estm.support_)