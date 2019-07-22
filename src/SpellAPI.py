import numpy as np 
import flask 
import io 
from flask import request, render_template
  
# Create Flask application and initialize Keras model 
app = flask.Flask(__name__) 
model = None
import torch

from SpellCorrector.Vocab import Voc
from SpellCorrector.DataPrepare import *
from SpellCorrector.Model import Seq2Seq,GreedySearchDecoder
from SpellCorrector import config
# device = torch.device("cpu")
device = torch.device("cpu")
loadFilename = config.evaluate['loadFilename']

def loadModel(modelFile):

    voc=Voc( config.data['corpus name'])
    if modelFile:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(modelFile,map_location='cpu')
        # checkpoint = torch.load(loadFilename)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']


    model=Seq2Seq(voc,device)

    print('Building encoder and decoder ...')

    if modelFile:
        model.embedding.load_state_dict(embedding_sd)

    if modelFile:
        model.encoder.load_state_dict(encoder_sd)
        model.decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    model.encoder = model.encoder.to(device)
    model.decoder = model.decoder.to(device)
    model.encoder.eval()
    model.decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(model.encoder, model.decoder,device)
    return model,searcher

model,searcher=loadModel(loadFilename)

@app.route('/predict')
def my_form():
    return '<form method="POST"><input name="text"><input type="submit"></form>' 
# Function to Load the model  
def load_model(): 
      
    # global variables, to be used in another function 
    pass
  
# Every ML/DL model has a specific format 
# of taking input. Before we can predict on 
# the input image, we first need to preprocess it. 
def prepare_text(text): 
    dataset=[]
    segments=text.split(" ")
    s=""
    for seg in segments:
        if len(s)+len(seg)>49:
            dataset.append(s)
            s=""
        s+=seg+" "
    if len(s)>10:
        dataset.append(s) 
  
    # return the processed image 
    return dataset
  
# Now, we can predict the results. 
@app.route("/predict", methods =["POST"]) 
def predict(): 
    data = {} # dictionary to store result 
    data["success"] = False
    # Check if image was properly sent to our endpoint 
    if flask.request.method == "POST": 
            result=""
            text=request.form['text']
            for d in prepare_text(text):
                result+=model.evaluateInput(model.encoder, model.decoder, searcher, model.voc,d)
                
            data["Input"]=text
            data["predictions"]=[{'Corrected Spelling':result}]
  
            data["success"] = True
  
    # return JSON response 
    return flask.jsonify(data) 
  
  
  
if __name__ == "__main__": 
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started")) 
    # load_model() 
    app.run(debug=True) 