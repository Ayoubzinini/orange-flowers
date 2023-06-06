from pandas import read_excel, DataFrame
from seaborn import boxplot, lineplot, set_theme, barplot
import pylab
from scipy.stats import shapiro, probplot
from scipy.signal import savgol_filter, detrend
from matplotlib.pyplot import show, rcParams, plot
from numpy import sqrt, mean
from statsmodels.multivariate.pca import PCA
from preproc_NIR import osc, msc, snv, simple_moving_average, centring, prep_log
db=read_excel("orange-flowers-data-mean.xlsx")
Y=db['Y']
X=db.drop(['Unnamed: 0','Y'],axis=1)
X=DataFrame(detrend(msc(savgol_filter(X,3,1,1))))
X.index=db['Unnamed: 0']
w,p = shapiro(Y)
if p>0.05:
  desicion="Normal"
elif p<0.05:
  desicion="Not Normal"
print('Quantile shapiro : {}\npropability shapiro : {}\ndesicion : {}'.format(w,p,desicion))
set_theme(style='darkgrid',rc={"axes.spines.right": False, "axes.spines.top": False})
rcParams['figure.figsize']=(12,8)
boxplot(Y)
show()
probplot(Y,dist="norm",plot=pylab)
show()
for i in X.index:
    plot(X.columns,X.loc[i,])
show()
for i in X.index:
    d=DataFrame()
    d['X']=mean(X,axis=0)
    d['Y']=X.loc[i,]
    lineplot(data=d,x='X',y='Y')
show()
"""
pc=PCA(X,X.shape[0],method='nipals')
for i in pc.loadings.columns:
    plot(pc.loadings.index,pc.loadings[i])
    show()
#"""