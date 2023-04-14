import sys
from os import rename, listdir, getcwd
from shutil import move, rmtree
ls=listdir(getcwd())
ls.remove('remane-them.py')
ls.remove('merge-them.py')
for i in ls:
    #"""
    ifls=listdir(getcwd()+'\\'+i)
    nls=[(i+'-'+j).replace(' ','-') for j in ifls]
    #"""
    for k,l in zip(ifls,nls):
        rename(getcwd()+'\\'+i+'\\'+k, getcwd()+'\\'+i+'\\'+l)
    for m in listdir(getcwd()+'\\'+i):
        move(getcwd()+'\\'+i+'\\'+m,getcwd()+'\\'+m)
    rmtree(getcwd()+'\\'+i)
    #"""
    #"""