import sys
from os import rename, listdir, getcwd
from pandas import read_csv, DataFrame
ls=listdir(getcwd())
ls.remove('remane-them.py')
ls.remove('merge-them.py')
db=DataFrame()
for i in ls:
    ifls=listdir(getcwd()+'\\'+i)
    b_spec=read_csv(getcwd()+'\\'+i+'\\'+ifls[len(ifls)-1],sep='\t')
    spec=b_spec["y_Axis:%Reflectance or Transmittance"]
    wl=b_spec["x_Axis:Wavelength (nm)"]
    db[i]=spec
db=db.T
db.columns=wl
db.to_excel("fleurs-orange-data-2021.xlsx")