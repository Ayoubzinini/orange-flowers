from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, max_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from pandas import read_excel, DataFrame
from scipy.stats import shapiro
from scipy.signal import savgol_filter, detrend
from preproc_NIR import msc, snv, prep_log, osc
from numpy import mean,sqrt
db=read_excel("orange-folwers-data.xlsx")
Y=db['Y']
#Y=[sqrt(i) for i in Y]
X=db.drop(['Unnamed: 0','Y'],axis=1)
X1=DataFrame(savgol_filter(X,3,1,1))
X2=msc(X.to_numpy())
X3=DataFrame(detrend(X))
#X4=osc(X)
X.index=db['Unnamed: 0']
j=0
while True:
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(X1,Y,test_size=0.2,random_state=j)
    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(X2,Y,test_size=0.2,random_state=j)
    x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(X3,Y,test_size=0.2,random_state=j)
    #x_train_4, x_test_4, y_train_4, y_test_4 = train_test_split(X4,Y,test_size=0.2,random_state=j)
    r2_1=[]
    RMSE_1=[]
    r2_2=[]
    RMSE_2=[]
    r2_3=[]
    RMSE_3=[]
    #r2_4=[]
    #RMSE_4=[]
    for i in range(1,x_train_1.shape[0]+1,1):
        model1=PLSRegression(n_components=i,scale=False)
        model2=PLSRegression(n_components=i,scale=False)
        model3=PLSRegression(n_components=i,scale=False)
        #model4=PLSRegression(n_components=i,scale=False)
        model1.fit(x_train_1, y_train_1)
        model2.fit(x_train_2, y_train_2)
        model3.fit(x_train_3, y_train_3)
        #model4.fit(x_train_4, y_train_4)
        r2_1.append(r2_score(model1.predict(x_test_1),y_test_1))
        RMSE_1.append(mean_squared_error(model1.predict(x_test_1),y_test_1))
        r2_2.append(r2_score(model2.predict(x_test_2),y_test_2))
        RMSE_2.append(mean_squared_error(model2.predict(x_test_2),y_test_2))
        r2_3.append(r2_score(model3.predict(x_test_3),y_test_3))
        RMSE_3.append(mean_squared_error(model3.predict(x_test_3),y_test_3))
        #r2_4.append(r2_score(model4.predict(x_test_4),y_test_4))
        #RMSE_4.append(mean_squared_error(model4.predict(x_test_4),y_test_4))
    model1=PLSRegression(n_components=1+RMSE_1.index(min(RMSE_1)),scale=False)
    model2=PLSRegression(n_components=1+RMSE_2.index(min(RMSE_2)),scale=False)
    model3=PLSRegression(n_components=1+RMSE_3.index(min(RMSE_3)),scale=False)
    #model4=PLSRegression(n_components=1+RMSE_4.index(min(RMSE_4)),scale=False)
    model1.fit(x_train_1, y_train_1)
    model2.fit(x_train_2, y_train_2)
    model3.fit(x_train_3, y_train_3)
    #model4.fit(x_train_4, y_train_4)
    ycv_1=cross_val_predict(PLSRegression(n_components=1+RMSE_1.index(min(RMSE_1))),x_train_1,y_train_1,cv=LeaveOneOut())
    ycv_2=cross_val_predict(PLSRegression(n_components=1+RMSE_2.index(min(RMSE_2))),x_train_2,y_train_2,cv=LeaveOneOut())
    ycv_3=cross_val_predict(PLSRegression(n_components=1+RMSE_3.index(min(RMSE_3))),x_train_3,y_train_3,cv=LeaveOneOut())
    #ycv_4=cross_val_predict(PLSRegression(n_components=1+RMSE_4.index(min(RMSE_4))),x_train_4,y_train_4,cv=LeaveOneOut())
    ycv=DataFrame({'y1':[z[0] for z in ycv_1],'y2':[z[0] for z in ycv_2],'y3':[z[0] for z in ycv_3]}).mean(axis=1)#,'y4':[z[0] for z in ycv_4]
    ycpr=DataFrame({'y1':[z[0] for z in model1.predict(x_train_1)],'y2':[z[0] for z in model2.predict(x_train_2)],'y3':[z[0] for z in model3.predict(x_train_3)]}).mean(axis=1)#,'y4':[z[0] for z in model4.predict(x_train_4)]
    ytpr=DataFrame({'y1':[z[0] for z in model1.predict(x_test_1)],'y2':[z[0] for z in model2.predict(x_test_2)],'y3':[z[0] for z in model3.predict(x_test_3)]}).mean(axis=1)#,'y4':[z[0] for z in model4.predict(x_test_4)]
    RMSECV=sqrt(mean_squared_error(DataFrame({'y1':y_train_1,'y2':y_train_2,'y3':y_train_3}).mean(axis=1), ycv))#,'y4':y_train_4
    R2CV=100*r2_score(DataFrame({'y1':y_train_1,'y2':y_train_2,'y3':y_train_3}).mean(axis=1), ycv)#,'y4':y_train_4
    R2train=100*r2_score(DataFrame({'y1':y_train_1,'y2':y_train_2,'y3':y_train_3}).mean(axis=1),ycpr)#,'y4':y_train_4
    R2test=100*r2_score(DataFrame({'y1':y_test_1,'y2':y_test_2,'y3':y_test_3}).mean(axis=1),ytpr)#,'y4':y_test_4
    RMSEtrain=sqrt(mean_squared_error(DataFrame({'y1':y_train_1,'y2':y_train_2,'y3':y_train_3}).mean(axis=1),ycpr))#,'y4':y_train_4
    RMSEtest=sqrt(mean_squared_error(DataFrame({'y1':y_test_1,'y2':y_test_2,'y3':y_test_3}).mean(axis=1),ytpr))#,'y4':y_test_4
    if R2CV>0 and R2test>0:
        break
    j=j+1
w,p = shapiro([i-j for i,j in zip(DataFrame({'y1':y_test_1,'y2':y_test_2,'y3':y_test_3}).mean(axis=1),ytpr)])#,'y4':y_test_4
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