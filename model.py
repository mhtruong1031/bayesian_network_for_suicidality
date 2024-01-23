from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import BIFWriter
from variables import relationships, training_data

model = BayesianNetwork() # .predict() for states .predict_probability() for probability
edges = []

for relation in relationships:
    input  = relation[0]
    output = relation[1]

    for i in input:
        for o in output:
            edges.append((i.name, o.name))

model.add_edges_from(edges)
model.fit(training_data)

BIFWriter(model).write_bif('model.bif')