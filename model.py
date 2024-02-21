import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import BIFWriter
from pgmpy.metrics import correlation_score
from pgmpy.metrics.bn_inference import BayesianModelProbability
from variables import relationships, training_data, testing_data

omitted_relationships = [
    ('q23_new', 'qn85'),
    ('qn66', 'qn85'),
    ('qnillict', 'qn85'),
    ('qn13_new', 'qn28'),
    ('qn94', 'qn28')
]

model = BayesianNetwork() # .predict() for states .predict_probability() for probability
edges = []

for relation in relationships:
    input  = relation[0]
    output = relation[1]

    for i in input:
        for o in output:
            edge = (i.name, o.name)
            if edge not in omitted_relationships:
                edges.append(edge)

model.add_nodes_from(['qn13_new', 'index', 'qn94'])
model.add_edges_from(edges)
model.fit(training_data)