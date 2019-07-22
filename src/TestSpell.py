import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import itertools
from SpellCorrector.Vocab import Voc
from SpellCorrector.DataPrepare import *
from SpellCorrector.Model import *
device = torch.device("cpu")
USE_CUDA = torch.cuda.is_available()
# device = torch.device("cuda" if USE_CUDA else "cpu")
loadFilename = "data/save/cb_model/data/questions/2-2_500/4000_checkpointNew.tar"
checkpoint_iter = 4000
clip = 50.0

learning_rate = 0.0003
decoder_learning_ratio = 5.0
n_iteration = 500
print_every = 10
save_every = 500
# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
voc=Voc( "data/questions")
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename,map_location='cpu')
    # checkpoint = torch.load(loadFilename)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_char, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_char, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
encoder.eval()
decoder.eval()
# test_inpt=["hw mch si teh pakring chrge","cn I gt a laon fr hbd"]
# test_output=["how much is the parking charge","can i get a loan for hdb"]
test_inpt=[line.rstrip('\n') for line in open("testX.txt")]      
test_output=[line.rstrip('\n') for line in open("testY.txt")]   
def findMismatch(inp,out):
    incorrectWords=0
    incorrectStatement=0
    for x,y in zip(inp,out):
        x=x.split(" ")
        y=y.split(" ")
        errors=0
        if len(x)!=len(y):
            incorrectWords+=len(y)
            incorrectStatement+=1
            continue
        for i in range(len(x)):
            if x[i].lower()!=y[i].lower():
                incorrectWords+=1
                errors=1
        if errors==1:
            errors=0
            incorrectStatement+=1
    return incorrectWords,incorrectStatement
# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder,device)

totalIncorrectWords,totalIncorrectStatements=findMismatch(test_inpt,test_output)
# Begin chatting (uncomment and run the following line to begin)
predicted=[]
for x in test_inpt:
    predicted.append(evaluateInput(encoder, decoder, searcher, voc,x))
predictedIncorrectWords,predictedIncorrectStatements=findMismatch(predicted,test_output)
print (predicted)
print ("Word accuracy is: ",1-(predictedIncorrectWords/totalIncorrectWords),"Sentence accuracy is:", 1-(predictedIncorrectStatements/totalIncorrectStatements))


