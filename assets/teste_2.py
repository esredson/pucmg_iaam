
import pandas as pd
dict = {
    'col1': ['1.1', '2.1', '3.1'],
    'col2': ['1.2', '2.2', '3.2'],
    'col3': ['1.3', '2.3', '3.3']
}
df = pd.DataFrame(dict)
df2 = df.loc[ df['col3'] == '3.3', ['col1', 'col2']]
print(df2)