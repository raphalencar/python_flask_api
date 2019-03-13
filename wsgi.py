import os
from run_keras_server import app as application 
from run_keras_server import load_model

print(" Loading keras model and Flask starting server... ")

input_shape = (155, 220, 3)
load_model(input_shape)
application.run()