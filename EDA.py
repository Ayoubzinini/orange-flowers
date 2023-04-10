from pandas import read_excel
from seaborn import boxplot
import pylab
from scipy.stats import shapiro, probplot
from matplotlib.pyplot import show
from numpy import sqrt
db=read_excel("fleurs-orange-data-2021.xlsx")
Y=db['Y']
X=db.drop(['Unnamed: 0','Y'],axis=1)
X.index=db['Unnamed: 0']
w,p = shapiro(Y)
if p>0.05:
  desicion="Normal"
elif p<0.05:
  desicion="Not Normal"
print('Quantile shapiro : {}\npropability shapiro : {}\ndesicion : {}'.format(w,p,desicion))
boxplot(Y)
show()
probplot(Y,dist="norm",plot=pylab)
show()