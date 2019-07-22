data={
    'sentence length':50,
    'corpus folder': '../corpus',
    'original data':'cleanHDB.txt',
    'mistakes folder':"../spelling-mistakes",
    
}
data['processed input file']="formatted"+data['original data']

model={
    'sentence length' :50, # maximum sentence length
    'PAD_token' : 0,  # Used for padding short sentences
    'SOS_token' : 1,  # Start-of-sentence token
    'EOS_token' : 2,  # End-of-sentence token
    'teacher_forcing_ratio' : 1.0,
    'hidden_size' : 500,
    'batch_size' : 64,
    'attn_model' : 'dot',
    'model_name':'cb_model',
    #'attn_model' : 'general',
    #'attn_model' : 'concat',
    'encoder_n_layers' : 4,
    'decoder_n_layers': 4,
    'dropout': 0.1,
    'clip': 50.0,
    'learning_rate': 0.0001,
    'decoder_learning_ratio': 5.0,
    'n_iteration': 400000,
    'print_every': 50000,
    'save_every': 50000,

}
evaluate={
    'loadFilename':"hdb.tar"

}