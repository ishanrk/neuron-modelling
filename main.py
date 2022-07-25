#implementing a probabilisitc graphical model using directed acyclic graphs 
#in bayesian networks


#library needed pgmpy

import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination

#made a model with pairwise dependencies and set up the structure
job_model= BayesianNetwork([('Difficulty','Grade'),
('STD_test_score','Grade'),
('Grade','Job')])

#setup conditional probability distributions

Difficulty_cpd = TabularCPD( 
    variable = 'Difficulty',
    variable_card=2,
    values = [[0.2],[0.8]] 
)

STD_test_score_cpd = TabularCPD(
    variable = 'STD_test_score',
    variable_card =2,
    values = [[0.7],[0.3]]
)

Job_cpd = TabularCPD(
    variable = 'Job',
    variable_card=2,
    values = [[0.95,0.5,0.8],
    [0.05,0.5,0.2]],
    evidence=['Grade'],
    evidence_card = [3]
)

Grade_cpd = TabularCPD(
    variable='Grade',
    variable_card=3,
    values = [[0.5,0.8,0.8,0.9],
    [0.3,0.15,0.1,0.08],
    [0.2,0.05,0.1,0.02]],

    evidence=['Difficulty','STD_test_score'],
    evidence_card=[2,2]
)



job_model.add_cpds(Difficulty_cpd,STD_test_score_cpd,Job_cpd,Grade_cpd)

#Showing struct of graph
print("CPD STRUCTURE OF GRAPH: ")
print(job_model.get_cpds())


print("ALL INDEPENDENCIES")
print(job_model.get_independencies())

print("ACTIVE NODE TRAIL NEEDED OUT OF Difficulty, Grade, STD_test_score,Job")
x = input("ENTER NODE ")
print(job_model.active_trail_nodes(x))


#used for inferring implicit CPDS
job_infer = VariableElimination(job_model)


prob_job = job_infer.query(variables=['Job'])
print("CPD OF JOB OFFER OVER ALL VARIABLES")



print(prob_job)

#CPD based on known knowledge

#probability of a bad test score getting a job

y = input('Enter what node you want to use as evidence for Job probability (Difficulty or STD_test_score): ')
z = input('Enter 0 for unfavourable interpreattion of node or 1 as favourable interpretation of node: ')

prob_bad_test = job_infer.query(
    variables = ['Job'], evidence = {y:z}
)

print("Probability of getting a job regardless of bad test scores")
print(prob_bad_test)


