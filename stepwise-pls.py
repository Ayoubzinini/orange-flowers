from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, max_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from pandas import read_excel, DataFrame
from scipy.stats import shapiro, f_oneway
from scipy.signal import savgol_filter, detrend
from preproc_NIR import msc, snv, prep_log
from numpy import mean,sqrt, std, median
from pysptools.spectro import convex_hull_removal
from dtwalign import dtw
db=read_excel("orange-flowers-data-mean.xlsx")
Y=db['Y']
X=db.drop(['Unnamed: 0','Y'],axis=1)
wl=X.columns
X=prep_log(X)
X=msc(X.to_numpy())
X=DataFrame(savgol_filter(X,3,1,1))
#X=DataFrame(detrend(X)) 
X.index=db['Unnamed: 0']
X.columns=wl
j=0
while True:
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=j)
    income_groups=[y_train,y_test]
    s,p=f_oneway(*income_groups)
    if p<0.05:
        break
    j+=1
estm=RFECV(PLSRegression(n_components=20,scale=False),cv=5)
estm.fit(x_train,y_train)
print(estm.support_)