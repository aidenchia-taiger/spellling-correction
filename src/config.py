
data={
    'max sentence length':50,
    'min sentence length': 1,
    'max incorrect generations per sentence': 20,
    'corpus': 'sample.txt',
    'training file path':'../corpus/sample.txt',
    'processed file path': '../corpus/formatted_sample.txt',
    'mistakes folder':"../spelling-mistakes",
    'confusion matrix dist path': '../spelling-mistakes/cm-distribution.csv',
    'delimiter': '\t'
}

model={
    'sentence length' :50, 
    'min sentence length': 1, 
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
    'loadFilename':"../Models/hdb.tar"

}