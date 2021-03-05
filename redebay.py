import numpy as np
import pandas as pd
from IPython.display import Image

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator

df = pd.read_csv('mushrooms2.csv')


model = BayesianModel([('cap-surface', 'odor'), ('spore-print-color', 'odor'), ('spore-print-color', 'gill-attachment'), ('odor', 'bruises'), ('odor', 'gill-size'),('odor', 'ring-type'),('odor', 'class'),('ring-type','class'), ('ring-type', 'bruises'), ('spore-print-color', 'class'), ('habitat', 'gill-attachment'), ('gill-color', 'ring-type'), ('gill-color', 'gill-size'), ('gill-color', 'class'), ('gill-color', 'gill-attachment')])

# mle = MaximumLikelihoodEstimator(model, df)

# print('cap-surface')
# print(mle.estimate_cpd('cap-surface')) 
# print('spore-print-color')
# print(mle.estimate_cpd('spore-print-color')) 
# print('habitat')
# print(mle.estimate_cpd('habitat')) 
# print('gill-color')
# print(mle.estimate_cpd('gill-color')) 
# print('odor')
# print(mle.estimate_cpd('odor')) 
# print('ring type')
# print(mle.estimate_cpd('ring-type')) 
# print('bruises')
# print(mle.estimate_cpd('bruises')) 
# print('gill size')
# print(mle.estimate_cpd('gill-size')) 
# print('gill attachment')
# print(mle.estimate_cpd('gill-attachment')) 
# print('class')
# print(mle.estimate_cpd('class')) 


cpd_cap_surface = TabularCPD(variable='cap-surface', variable_card=3, values= [[0.286066], [0.314623], [0.399311]])

cpd_spore_print_color = TabularCPD(variable='spore-print-color', variable_card=3, 
                     values= [[0.242245], [0.463811], [0.293944]])


cpd_habitat = TabularCPD(variable='habitat', variable_card=4, 
                    values= [[0.387494], [0.264402], [0.207287], [0.140817]])

cpd_gill_color = TabularCPD(variable='gill-color', variable_card=4, 
                    values= [[0.212703], [0.129], [0.51034 ], [0.147957]])

cpd_odor = TabularCPD(variable='odor', variable_card=3, 
                   values= [[0.0 ,0.46551724137931033, 0.0 ,0.0 ,0.3116883116883117 , 0.2553191489361702, 0.0 ,0.4462809917355372 , 0.26865671641791045],
                            [0.9032258064516129, 0.4827586206896552 , 1.0 ,0.47619047619047616, 0.4025974025974026 , 0.23404255319148937, 0.6 ,0.35537190082644626, 0.16044776119402984],
                            [0.0967741935483871, 0.05172413793103448, 0.0 ,0.5238095238095238, 0.2857142857142857 , 0.5106382978723404, 0.4 ,0.19834710743801653, 0.5708955223880597]],
                  evidence=['cap-surface', 'spore-print-color'],
                  evidence_card=[3,3])

cpd_ring_type = TabularCPD(variable='ring-type', variable_card=3, 
                   values= [[1.0, 0.3333333333333333, 1.0, 0.3333333333333333, 0.2696629213483146, 0.0, 0.0, 0.3353174603174603  , 0.0, 0.0, 0.225, 0.0],
                            [0.0, 0.3333333333333333, 0.0, 0.3333333333333333, 0.0     , 0.0, 0.8709677419354839 , 0.017857142857142856, 0.028037383177570093, 0.0, 0.015  , 0.058823529411764705],
                            [0.0, 0.3333333333333333, 0.0, 0.3333333333333333, 0.7303370786516854, 1.0, 0.12903225806451613, 0.6468253968253969  , 0.9719626168224299  , 1.0, 0.76   , 0.9411764705882353]],
                  evidence=['gill-color', 'odor'],
                  evidence_card=[4,3])

cpd_bruises = TabularCPD(variable='bruises', variable_card=2, 
                   values= [[1.0 , 1.0, 0.0 , 0.816793893129771 , 1.0      , 0.24342105263157895 , 1.0 , 1.0      , 0.15384615384615385],
                            [0.0 , 0.0, 1.0 , 0.183206106870229 , 0.0, 0.756578947368421   , 0.0 , 0.0      , 0.8461538461538461]],
                  evidence=['odor', 'ring-type'],
                  evidence_card=[3,3])

cpd_gill_size = TabularCPD(variable='gill-size', variable_card=2, 
                   values= [[0.0 ,0.5 ,0.0 ,0.5 ,0.9662921348314607 , 0.5714285714285714, 1.0      ,0.9444444444444444, 0.5264797507788161 ,1.0 ,0.87,0.6862745098039216],
                            [1.0 ,0.5 ,1.0 ,0.5 ,0.033707865168539325 ,0.42857142857142855 ,0.0      ,0.05555555555555555 ,0.4735202492211838 ,0.0 ,0.13,0.3137254901960784]],
                  evidence=['gill-color', 'odor'],
                  evidence_card=[4,3])

cpd_gill_attachment = TabularCPD(variable='gill-attachment', variable_card=2, 
                   values= [[0.5,0.5 ,0.0,0.5,0.5 ,0.5,0.5,0.5 ,0.0,0.5,0.5 ,0.0,0.0,0.0 ,0.5,0.0,0.0 ,0.5,0.21052631578947367  , 0.4444444444444444    , 0.5,0.0,0.0 ,0.5,0.0,0.0 ,0.5,0.0,0.0 ,0.0,0.1951219512195122   , 0.27586206896551724   , 0.0,0.0,0.0 ,0.5,0.0,0.0 ,0.15517241379310345  , 0.0,0.0 ,0.0,0.0,0.0 ,0.0,0.0,0.0 ,0.0],
                            [0.5,0.5 ,1.0,0.5,0.5 ,0.5,0.5,0.5 ,1.0,0.5,0.5 ,1.0,1.0,1.0 ,0.5,1.0,1.0 ,0.5,0.7894736842105263   , 0.5555555555555556    , 0.5,1.0,1.0 ,0.5,1.0,1.0 ,0.5,1.0,1.0 ,1.0,0.8048780487804879   , 0.7241379310344828    , 1.0,1.0,1.0 ,0.5,1.0,1.0 ,0.8448275862068966   , 1.0,1.0 ,1.0,1.0,1.0 ,1.0,1.0,1.0 ,1.0]],
                  evidence=['gill-color', 'habitat', 'spore-print-color'],
                  evidence_card=[4,4,3])

cpd_class = TabularCPD(variable='class', variable_card=2, 
                values= [[0.5 ,0.5,0.0 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.0 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,1.0 ,1.0 ,0.5 ,0.5 ,0.5 ,0.5 ,1.0 ,1.0 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.6666666666666666   ,0.6666666666666666   ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.0 ,0.5 ,0.5 ,0.0 ,0.5 ,1.0 ,1.0 ,0.96,0.5 ,1.0 ,0.5 ,1.0 ,0.9215686274509803   ,1.0 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.0 ,0.5641025641025641   ,0.5641025641025641   ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.0 ,0.5 ,0.5 ,0.5 ,0.8 ,0.5 ,1.0 ,0.5 ,1.0 ,0.9 ,0.9473684210526315   ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.0 ,0.7777777777777778   ,0.7777777777777778   ,0.5],
                        [0.5 ,0.5 ,1.0 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,1.0 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.0 ,0.0 ,0.5 ,0.5 ,0.5 ,0.5 ,0.0 ,0.0 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.3333333333333333   ,0.3333333333333333   ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,1.0 ,0.5 ,0.5 ,1.0 ,0.5 ,0.0 ,0.0 ,0.04,0.5 ,0.0 ,0.5 ,0.0 ,0.0784313725490196   ,0.0 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,1.0 ,0.4358974358974359   ,0.4358974358974359   ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,1.0 ,0.5 ,0.5 ,0.5 ,0.2 ,0.5 ,0.0 ,0.5 ,0.0 ,0.1 ,0.05263157894736842  ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,1.0 ,0.2222222222222222   ,0.2222222222222222   ,0.5]], 
                evidence=['odor', 'ring-type', 'spore-print-color', 'gill-color'],
                evidence_card=[3,3,3,4])
        
model.add_cpds(cpd_cap_surface, cpd_spore_print_color, cpd_habitat, cpd_gill_color, cpd_odor, cpd_ring_type, cpd_bruises, cpd_gill_size, cpd_gill_attachment, cpd_class)
model.check_model()
model.get_cpds()

# # print("Nodes: ", model.nodes())
# # print("Edges: ", model.edges())

# # print(model.get_cpds('spore-print-color')) 
# # print(model.get_cpds('odor')) 
# # print(model.get_cpds('class')) 

# # model.fit(data=df, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=8124)
# # print(model.get_cpds('class'))

# # model_infer = VariableElimination(model)
# # q = model_infer.query(variables=['class'], evidence={'6': 'a'})
# # print(q)
# model_infer = VariableElimination(model)
# x = model_infer.query(variables=['income'], evidence={'odor':0})
# print(x)