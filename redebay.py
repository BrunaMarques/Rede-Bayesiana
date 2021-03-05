import numpy as np
import pandas as pd
from IPython.display import Image

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator

df = pd.read_csv('teste.csv')

model = BayesianModel([('cap-surface', 'odor'), ('spore-print-color', 'odor'), ('spore-print-color', 'gill-attachment'), ('odor', 'bruises'), ('odor', 'gill-size'),('odor', 'ring-type'),('odor', 'class'),('ring-type','class'), ('ring-type', 'bruises'), ('spore-print-color', 'class'), ('habitat', 'gill-attachment'), ('gill-color', 'ring-type'), ('gill-color', 'gill-size'), ('gill-color', 'class'), ('gill-color', 'gill-attachment')])

# mle = MaximumLikelihoodEstimator(model, df)
# print(mle.estimate_cpd('cap-surface')) 
# print(mle.estimate_cpd('spore-print-color')) 
# print(mle.estimate_cpd('habitat')) 
# print(mle.estimate_cpd('gill-color')) 
# print(mle.estimate_cpd('odor')) 
# print(mle.estimate_cpd('ring-type')) 
# print(mle.estimate_cpd('bruises')) 
# print(mle.estimate_cpd('gill-size')) 
# print(mle.estimate_cpd('gill-attachment')) 
# print(mle.estimate_cpd('class')) 

###### acho q ta errado, não é as colunas da tabela, é o cpd de cada coluna? tem que ser número


cpd_cap_surface = TabularCPD(variable='cap-surface', variable_card=1, values= [[1]])

cpd_spore_print_color = TabularCPD(variable='spore-print-color', variable_card=2, 
                     values= [[0.5], [0.5]])


cpd_habitat = TabularCPD(variable='habitat', variable_card=2, 
                    values= [[0.5], [0.5]])

cpd_gill_color = TabularCPD(variable='gill-color', variable_card=1, 
                    values= [[1]])

cpd_odor = TabularCPD(variable='odor', variable_card=2, 
                   values= [[0,1], [1,0]],
                  evidence=['cap-surface', 'spore-print-color'],
                  evidence_card=[1,2])

cpd_ring_type = TabularCPD(variable='ring-type', variable_card=1, 
                   values= [[1,1]],
                  evidence=['gill-color', 'odor'],
                  evidence_card=[1,2])

cpd_bruises = TabularCPD(variable='bruises', variable_card=1, 
                   values= [[1,1]],
                  evidence=['odor', 'ring-type'],
                  evidence_card=[2,1])

cpd_gill_size = TabularCPD(variable='gill-size', variable_card=2, 
                   values= [[1,0], [0,1]],
                  evidence=['gill-color', 'odor'],
                  evidence_card=[1,2])

cpd_gill_attachment = TabularCPD(variable='gill-attachment', variable_card=1, 
                   values= [[1,1,1,1]],
                  evidence=['gill-color', 'habitat', 'spore-print-color'],
                  evidence_card=[1,2,2])

cpd_class = TabularCPD(variable='class', variable_card=2, 
                   values= [[0.5,1,0,0.5], [0.5,0,1,0.5]],
                  evidence=['odor', 'ring-type', 'spore-print-color', 'gill-color'],
                  evidence_card=[2,1,2,1])
        
model.add_cpds(cpd_cap_surface, cpd_spore_print_color, cpd_habitat, cpd_gill_color, cpd_odor, cpd_ring_type, cpd_bruises, cpd_gill_size, cpd_gill_attachment, cpd_class)
model.check_model()
model.get_cpds()

print(model.get_cpds('spore-print-color')) 
print(model.get_cpds('odor')) 
print(model.get_cpds('class')) 