import numpy as np
import pandas as pd

a = [['12', '10'], ['11', '13']]
b = [1, 2]
df=pd.DataFrame(list(zip(a, b)))
print(df.loc[:,0].tolist())
print(a)
