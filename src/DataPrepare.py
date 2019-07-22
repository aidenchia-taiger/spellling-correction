import csv
import random
import re
import os
import unicodedata
from io import open
import itertools
import spacy
import tqdm
import pandas as pd
from pdb import set_trace
from numpy.random import choice as random_choice, randint as random_randint
import random
from Vocab import Vocab
import torch
import config


class Data:
    def __init__(self):   
        self.MAX_LENGTH = config.data['max sentence length']
        self.MIN_LENGTH = config.data['min sentence length']
        self.delimiter = config.data['delimiter']
        self.train_file = config.data['training file path']
        self.data_file = config.data['processed file path']
        self.mistakeDir = config.data['mistakes folder']
        self.cmDist = config.data['confusion matrix dist path']
        self.maxGenerations = config.data['max incorrect generations per sentence']
        self.nlp=spacy.load("en")
        self.voc = Vocab()
        self.save_dir=None
        [print('[INFO] {}: {}'.format(k,v)) for k, v in dict(vars(self)).items()]

    def load_spelling_mistakes(self):
        spelling_mistakes = {}
        files = os.listdir(self.mistakeDir)
        for mistake_file in files:
            file_path = os.path.join(self.mistakeDir, mistake_file)
            lines = open(file_path).readlines()
            for line in lines:
                key = line.split(":")[0].strip().lower()
                values = set([word.strip().lower() for word in line.split(":")[1].split()])
                if key in spelling_mistakes:
                    spelling_mistakes[key].union(values)
                else:
                    spelling_mistakes[key] = values
            return spelling_mistakes

    def add_mistakes(self,text, tokenized=True):
        spelling_mistake_tokens = []
        if tokenized:
            tokens = text.split()
        else:
            if self.nlp is None:
                self.nlp = spacy.load("en")
            doc = self.nlp(text)
            tokens = [token.text for token in doc]

        common_mistakes=self.load_spelling_mistakes()
        for token in tokens:
            if token.lower() in common_mistakes:
                spelling_mistake_tokens.append(random.choice(list(common_mistakes[token.lower()])))
            else:
                spelling_mistake_tokens.append(token.lower())
        return " ".join(spelling_mistake_tokens)

    def swap_misclassified_chars(self, a_string, nIterations=5):
        """Given a string (word), returns a list of strings of length nIterations,
        where characters in each string is replaced by another character 
        with probability according to confusion matrix distribution
        """
        mistakesDist = pd.read_csv(self.cmDist, index_col=0).T
        population = list(mistakesDist.columns.values)
        new_string = ''
        incorrectVersions = []
        for _ in range(nIterations):
            for char in a_string:
                if char not in population: # char could be ' ' or other special characters like "!"
                    new_string += char
                    continue
                
                # Randomly replace existing character with new character according to distribution
                new_char = random_choice(population, size=None, replace=True, p=list(mistakesDist[char]))        

                #if new_char != char:
                #    print('[INFO] Replacing {} with {}'.format(char, new_char))

                new_string += new_char
            incorrectVersions.append(new_string)
            new_string = ''
        return incorrectVersions


    def add_noise_to_string(self,a_string):
        """Adds aritificial random noise to a string, returns a list of strings with noise added"""
        CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .")
        incorrectVersions=[]
        origString=a_string
        for i in range(random.randrange(1,4)):
            a_string=origString
            onehop=random.randrange(1,3)
            for _ in range(onehop):
                j=random.randrange(1,5)
                if j==1 and len(a_string)>0:
                    # Replace a character with a random character
                    random_char_position = random_randint(len(a_string))
                    a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position + 1:]
                elif j==2 and len(a_string)>0:
                    # Delete a character
                    random_char_position = random_randint(len(a_string))
                    a_string = a_string[:random_char_position] + a_string[random_char_position + 1:]

                elif j==3:
                    # Add a random character
                    if len(a_string)>0:
                        random_char_position = random_randint(len(a_string))
                        a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position:]
                    else:
                        a_string=random_choice(CHARS[:-1])
                elif len(a_string)>1:
                    # Transpose 2 characters
                    random_char_position = random_randint(len(a_string) - 1)
                    a_string = (a_string[:random_char_position] + a_string[random_char_position + 1] + a_string[random_char_position] +
                                a_string[random_char_position + 2:])
                incorrectVersions.append(a_string)
        return incorrectVersions


    def printLines(self,file, n=10):
        with open(file, 'rb') as datafile:
            lines = datafile.readlines()
        for line in lines[:n]:
            print(line)

    def unicodeToAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    def normalizeString(self,s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    def modifyString(self, sentence):
        new_sentence = ''
        for char in sentence:
            # these special characters are treated as the same class as their upper-case versions
            if char in ['w', 's', 'o', 'c', 'u', 'i', 'm', 'p', 'v', 'w', 'x', 'y', 'z']:
                new_sentence += char.upper()
            else:
                new_sentence += char
        return new_sentence

    # Read query/response pairs and return a voc object
    def readVocs(self,datafile=None):
        datafile=self.data_file if datafile is None else datafile
        print("Reading lines...")
        # Read the file and split into lines
        lines = open(datafile, encoding='utf-8').\
            read().strip().split('\n')
        # Split every line into pairs
        pairs = [[s for s in l.split('\t')] for l in lines]
        
        return pairs

    def split_sentences(self,text):
            sentences = []
            for sent in self.nlp(text).sents:
                sentences.append(sent.text)
            return sentences

    # Splits each line of the file into a dictionary of fields
    def read_data(self,train=True):
            print("[INFO] Reading data ... ")
            dataset = []
            if train:
                lines = open(self.train_file).readlines()
            else:
                lines = open(self.test_file).readlines()
            l=min(len(lines),500000)
            for line in tqdm.tqdm(lines[:l]):
                if line.strip() and not line.strip().startswith("="):
                    sentences = self.split_sentences(line.strip())
                    for sentence in sentences:
                        #sentence = self.normalizeString(sentence)
                        sentence = self.modifyString(sentence)
                        if len(sentence.split()) > self.MIN_LENGTH:
                            dataset.append(sentence)
            return dataset



    def create_pairs(self,dataset):
        inputLines=[]
        outputLines=[]

        for line in dataset:
            incorrectSentences=[]
            minWords=min(len(line.split(" ")),3)
            wordList=random_choice(line.split(" "),minWords,False)
            for word in line.split(" "):
                if word in wordList:
                    #incorrectWords=self.add_noise_to_string(word)
                    incorrectWords=self.swap_misclassified_chars(word)
                else:
                    incorrectWords=[word]

                if len(incorrectSentences)==0:
                    incorrectSentences=incorrectWords
                else:
                    newSentences=[]
                    for x in itertools.product(incorrectSentences,incorrectWords):
                        newSentences.append(' '.join(x))
                    incorrectSentences=list(set(newSentences))
                
            #incorrectSentences = [self.add_mistakes(text=sent, tokenized=True) for sent in incorrectSentences]

            random.shuffle(incorrectSentences)
            
            # Cap max no. of incorrect sentences generated per sentence
            if len(incorrectSentences) > self.maxGenerations:
                incorrectSentences=incorrectSentences[:self.maxGenerations]

            # Cap each sentence till max length
            input_lines = [sent[:self.MAX_LENGTH] for sent in incorrectSentences]
            output_lines = [line[:self.MAX_LENGTH]]*len(incorrectSentences)
            inputLines+=input_lines
            outputLines+=output_lines

        # zip and sort according to length for buckecting
        lines = zip(inputLines,outputLines)
        # lines = sorted(lines, key=lambda x: len(x[1]), reverse=True)
        lines=list(lines)
        random.shuffle(lines)
        return lines

    # Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
    def filterPair(self,p):
        # Input sequences need to preserve the last word for EOS token
        return len(p[0].split(' ')) < self.MAX_LENGTH and len(p[1].split(' ')) < self.MAX_LENGTH

    # Filter pairs using filterPair condition
    def filterPairs(self,pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    # Using the functions defined above, return a populated voc object and pairs list
    def loadPrepareData(self, save_dir):
        print("Start preparing training data ...")
        pairs = self.readVocs()
        print("Read {!s} sentence pairs".format(len(pairs)))
        pairs = self.filterPairs(pairs)
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))
        print("[INFO] Counting words...")
        for pair in pairs:
            self.voc.addSentence(pair[0])
            self.voc.addSentence(pair[1])
        print("[INFO] Counted words:", self.voc.num_char)
        return pairs

    def trimRareChars(self,pairs, MIN_COUNT):
        # Trim words used under the MIN_COUNT from the voc
        self.voc.trim(MIN_COUNT)
        # Filter out pairs with trimmed words
        keep_pairs = []
        for pair in pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True
            # Check input sentence
            for char in input_sentence:
                if char not in self.voc.char2index:
                    keep_input = False
                    break
            # Check output sentence
            for char in output_sentence:
                if char not in self.voc.char2index:
                    keep_output = False
                    break

            # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
            if keep_input and keep_output:
                keep_pairs.append(pair)

        print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
        return keep_pairs

    
    def formatDataset(self,datafile=None):
        datafile=self.data_file if datafile is None else datafile
        print("\nProcessing corpus...")
        lines = self.read_data()
        
        print("\nWriting newly formatted file...")
        with open(datafile, 'w', encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter=self.delimiter, lineterminator='\n')
            for pair in self.create_pairs(lines):
                writer.writerow(pair)
        print("\nSample lines from file:")
        self.printLines(datafile)
        
        self.save_dir = os.path.join("data", "save")
        pairs = self.loadPrepareData(self.save_dir)
        print("\npairs: ['Incorrect', 'Ground Truth']")
        
        for pair in pairs[:10]:
            print(pair)
        
        return pairs

if __name__ == "__main__":
    data = Data()



