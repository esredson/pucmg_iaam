from tkinter.ttk import Separator
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('assets/tmp_consulta.csv', sep='|', parse_dates=['dt'])
df = df.reset_index(drop=True)
df.plot(y='tmp', x='dt')
plt.show()
a = 0