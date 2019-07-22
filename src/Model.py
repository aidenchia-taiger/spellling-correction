import random
import json
import os
from pathlib import Path
import spacy
import tqdm
from numpy.random import choice as random_choice, randint as random_randint
import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import itertools
from Vocab import Vocab
from DataPrepare import Data
import config

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths,enforce_sorted=False)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder,device=None):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA and device is None else "cpu")
        #self.device =  torch.device("cpu")
        self.SOS_token= 1


    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self.device, dtype=torch.long) * self.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


class Seq2Seq:
    def __init__(self,voc,device=None):
        # device=torch.device("cpu")
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA and device is None else "cpu")
        self.MAX_LENGTH = config.model['sentence length']
        self.PAD_token = config.model['PAD_token']
        self.SOS_token = config.model['SOS_token']
        self.EOS_token = config.model['EOS_token']
        self.teacher_forcing_ratio = config.model['teacher_forcing_ratio']
        self.hidden_size = config.model['hidden_size']
        self.batch_size = config.model['batch_size']
        self.attn_model = config.model['attn_model']
        self.model_name=config.model['model_name']
        #self.attn_model = 'config.model['attn_model']
        #self.attn_model = config.model['attn_model']
        self.voc=voc
        self.encoder_n_layers = config.model['encoder_n_layers']
        self.decoder_n_layers = config.model['decoder_n_layers']
        self.dropout = config.model['dropout']
        self.embedding=nn.Embedding(self.voc.num_char, self.hidden_size)
        self.encoder=EncoderRNN(self.hidden_size, self.embedding, self.encoder_n_layers, self.dropout).to(self.device)
        self.decoder=LuongAttnDecoderRNN(self.attn_model, self.embedding, self.hidden_size, self.voc.num_char, self.decoder_n_layers, self.dropout).to(self.device)

        
    
    def indexesFromSentence(self,voc, sentence):
        return [voc.char2index[char] for char in sentence] + [self.EOS_token]


    def zeroPadding(self,l, fillvalue=None):
        if fillvalue is None:
            fillvalue=self.PAD_token
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def binaryMatrix(self,l, value=None):
        if value is None:
            value=self.PAD_token
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == self.PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    # Returns padded input sequence tensor and lengths
    def inputVar(self,l, voc):
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths

    # Returns padded target sequence tensor, padding mask, and max target length
    def outputVar(self,l, voc):
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        mask = self.binaryMatrix(padList)
        mask = torch.ByteTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, max_target_len

    # Returns all items for a given batch of pairs
    def batch2TrainData(self,voc, pair_batch):
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = self.inputVar(input_batch, voc)
        output, mask, max_target_len = self.outputVar(output_batch, voc)
        return inp, lengths, output, mask, max_target_len

    def maskNLLLoss(self,inp, target, mask):
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(self.device)
        return loss, nTotal.item()

    def train(self,input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
            encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=0):
        if max_length==0:
            max_length=self.MAX_LENGTH
        # Zero gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Set device options
        input_variable = input_variable.to(self.device)
        lengths = lengths.to(self.device)
        target_variable = target_variable.to(self.device)
        mask = mask.to(self.device)

        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[self.SOS_token for _ in range(batch_size)]])
        decoder_input = decoder_input.to(self.device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, nTotal = self.maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(self.device)
                # Calculate and accumulate loss
                mask_loss, nTotal = self.maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

        # Adjust model weights
        encoder_optimizer.step()
        decoder_optimizer.step()

        return sum(print_losses) / n_totals

    def trainIters(self,  pairs, encoder_optimizer, decoder_optimizer, save_dir, n_iteration, print_every, save_every, clip, corpus_name, loadFilename):
        
        # Load batches for each iteration
        training_batches = [self.batch2TrainData(self.voc, [random.choice(pairs) for _ in range(self.batch_size)])
                        for _ in range(n_iteration)]

        # Initializations
        print('Initializing ...')
        start_iteration = 1
        print_loss = 0
        if loadFilename:
            start_iteration = checkpoint['iteration'] + 1

        # Training loop
        print("Training...")
        for iteration in range(start_iteration, n_iteration + 1):
            training_batch = training_batches[iteration - 1]
            # Extract fields from batch
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            # Run a training iteration with batch
            loss = self.train(input_variable, lengths, target_variable, mask, max_target_len, self.encoder,
                        self.decoder, self.embedding, encoder_optimizer, decoder_optimizer, self.batch_size, clip)
            print_loss += loss

            # Print progress
            if iteration % print_every == 0:
                print_loss_avg = print_loss / print_every
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
                print_loss = 0

            # Save checkpoint
            if (iteration % save_every == 0):
                directory = os.path.join(save_dir, self.model_name, corpus_name, '{}-{}_{}'.format(self.encoder_n_layers, self.decoder_n_layers, self.hidden_size))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save({
                    'iteration': iteration,
                    'en': self.encoder.state_dict(),
                    'de': self.decoder.state_dict(),
                    'en_opt': encoder_optimizer.state_dict(),
                    'de_opt': decoder_optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': self.voc.__dict__,
                    'embedding': self.embedding.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpointLatest')))
    
    def training(self,trainData,corpus_name,save_dir):
        # Configure training/optimization
        clip = config.model['clip']

        learning_rate = config.model['learning_rate']
        decoder_learning_ratio = config.model['decoder_learning_ratio']
        n_iteration = config.model['n_iteration']
        print_every = config.model['print_every']
        save_every = config.model['save_every']

        # Ensure dropout layers are in train mode
        self.encoder.train()
        self.decoder.train()

        # Initialize optimizers
        print('Building optimizers ...')
        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
        
        # Run training iterations
        print("Starting Training!")
        
        self.trainIters(trainData, encoder_optimizer, decoder_optimizer,
                save_dir, n_iteration,
                print_every, save_every, clip, corpus_name, None)



    def evaluate(self,encoder, decoder, searcher, voc, sentence, max_length=0):
        if max_length==0:
            max_length=self.MAX_LENGTH
        ### Format input sentence as a batch
        # words -> indexes
        indexes_batch = [self.indexesFromSentence(voc, sentence)]
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(self.device)
        lengths = lengths.to(self.device)
        # Decode sentence with searcher
        tokens, scores = searcher(input_batch, lengths, max_length)
        # indexes -> words
        decoded_words = [voc.index2char[token.item()] for token in tokens]
        return decoded_words


    def evaluateInput(self,encoder, decoder, searcher, voc,inputData=""):
        input_sentence = ''
        if inputData=="":
            while(1):
                try:
                    # Get input sentence
                    input_sentence = input('> ')
                    # Check if it is quit case
                    if input_sentence == 'q' or input_sentence == 'quit': break
                    # Normalize sentence

                    input_sentence = Data().normalizeString(input_sentence)
                    # Evaluate sentence
                    output_words = self.evaluate(encoder, decoder, searcher, voc, input_sentence)
                    # Format and print response sentence
        #             output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                    output_words=output_words[:output_words.index('EOS')]
                    print('Bot:', ' '.join(output_words))

                except KeyError:
                    print("Error: Encountered unknown word.")
        else:
            input_sentence=inputData
            try:
            # Normalize sentence
                input_sentence = Data().normalizeString(input_sentence)
                # Evaluate sentence
                output_words = self.evaluate(encoder, decoder, searcher, voc, input_sentence)
                # Format and print response sentence
    #             output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                def getIndex(s,c):
                    r=55
                    try:
                        r=s.index(c)
                    except:
                        pass
                    
                    return r
                
                output_words=''.join(output_words)
                last=min(getIndex(output_words,"EOS"),getIndex(output_words,"PAD"),51)
                
                return output_words[:last]

            except KeyError:
                return "Error"

    def testing(self,testPairs):
        self.encoder.eval()
        self.decoder.eval()

        # Initialize search module
        searcher = GreedySearchDecoder(self.encoder, self.decoder)

        # Begin chatting (uncomment and run the following line to begin)
        acc=0
        for test in testPairs:
            x,y=test
            pred=self.evaluateInput(self.encoder, self.decoder, searcher, self.voc,x)
            # print ("X= ",x,"\nPred= ",pred,"\nY= ",y)
            if pred==y:
                acc+=1

        print (acc/len(testPairs))
