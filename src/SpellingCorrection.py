import torch
from DataPrepare import Data
from Model import Seq2Seq
import config

class SpellingCorrection:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.corpus_name = config.data['corpus']
        self.data=Data()
    
if __name__=="__main__":
    sc=SpellingCorrection()
    pairs=sc.data.formatDataset()
    print('[INFO] Building encoder and decoder ...')
    model=Seq2Seq(sc.data.voc)
    print('[INFO] Models built and ready to go!')
    trainPairs=pairs[:int(0.8*len(pairs))]
    testPairs=pairs[int(0.8*len(pairs)):]
    model.training(trainPairs,sc.corpus_name,sc.data.save_dir)
    model.testing(testPairs)
    