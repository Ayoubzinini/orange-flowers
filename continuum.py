%matplotlib nbagg
from pysptools import spectro as sp
import pandas as pd
import matplotlib.pyplot as plt
src = r"N:\E01_4m try.xlsx"

df = pd.read_excel(src, sheetname='data')

wvl = df.columns.tolist() #the wavelength must be a list
pixel = list(df.iloc[0,:]) # the spectrum must be a list
CR = sp.convex_hull_removal(pixel,wvl)

plt.plot(wvl,CR[0])