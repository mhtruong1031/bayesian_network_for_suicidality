import pandas as pd
from pgmpy.readwrite import BIFReader
from pgmpy.metrics import structure_score
from variables import training_data, testing_data

model = BIFReader('model.bif').get_model()

print(structure_score(model, training_data))