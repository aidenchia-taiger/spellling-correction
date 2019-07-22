# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Vocab:
    def __init__(self):
        self.trimmed = False
        self.char2index = {}
        self.char2count = {}
        self.index2char = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_char = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for char in sentence:
            self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.num_char
            self.char2count[char] = 1
            self.index2char[self.num_char] = char
            self.num_char += 1
        else:
            self.char2count[char] += 1

    # Remove characters below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_chars = []

        for k, v in self.char2count.items():
            if v >= min_count:
                keep_chars.append(k)

        print('keep_chars {} / {} = {:.4f}'.format(
            len(keep_chars), len(self.char2index), len(keep_chars) / len(self.char2index)
        ))

        # Reinitialize dictionaries
        self.char2index = {}
        self.char2count = {}
        self.index2char = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_char = 3 # Count default tokens

        for char in keep_chars:
            self.addChar(char)