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

'''predict_data = {
    'qn87': 1.0,
    'qn94': 1.0,
    'q23_new': 1.0,
    'qn22_new': 1.0,
    'q98': 2.0,
    'qn93': 2.0,
    'qn85': 2.0,
    'q25': 2.0,
    'qn66': 2.0,
    'qn67_new': 2.0,
    'qnillict': 1.0,
    'qn96': 1.0,
    'qn13_new': 1.0,
    'qn16': 1.0,
    'q18': 1.0,
    'q27': 1.0
}
predict_data_df = pd.DataFrame(data=predict_data, index=[0])
print(predict_data_df)
print(model.predict_probability(predict_data_df)) # TO DO: only predicts properly in model.py? find out why

BIFWriter(model).write_bif('model.bif')'''