import sys
from os import rename, listdir, getcwd
from shutil import rmtree
from pandas import read_csv, DataFrame
ls=listdir(getcwd())
ls.remove('remane-them.py')
ls.remove('merge-them.py')
db=DataFrame()
for i in ls:
    ifls=listdir(getcwd()+'\\'+i)
    f_spec=DataFrame()
    for j in ifls:
        b_spec=read_csv(getcwd()+'\\'+i+'\\'+j,sep='\t')
        spec=b_spec["y_Axis:%Reflectance or Transmittance"]
        f_spec[j]=spec
    wl=b_spec["x_Axis:Wavelength (nm)"]
    db[i]=f_spec.T.mean()
    rmtree(getcwd()+'\\'+i)
db=db.T
db.columns=wl
db.to_excel("fleurs-orange-data-2021.xlsx")