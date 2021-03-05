import numpy as np
import pandas as pd
from IPython.display import Image

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

df = pd.read_csv('mushrooms.csv')
print(df['class'])

model = BayesianModel([('cap-surface', 'odor'), ('spore-print-color', 'odor'), ('spore-print-color', 'gill-attachment'), ('odor', 'bruises'), ('odor', 'gill-size'),('odor', 'ryng-type'),('odor', 'classe'),('ryng-type','classe'), ('ryng-type', 'bruises'), ('spore-print-color', 'classe'), ('habitat', 'gill-attachment'), ('gill-color', 'ryng-type'), ('gill-color', 'gill-size'), ('gill-color', 'classe'), ('gill-color', 'gill-attachment')])
