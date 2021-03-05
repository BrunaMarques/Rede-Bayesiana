import numpy as np
import pandas as pd
from IPython.display import Image

df = pd.read_csv('mushrooms.csv')
print(df['class'])