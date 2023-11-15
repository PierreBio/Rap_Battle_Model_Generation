import os

import math
import numpy as np 
import time

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')

if IN_COLAB:
    ROOT_PATH = '/content/drive/MyDrive/Colab Notebooks/'
else:
    ROOT_PATH = './'
CHECKPOINT_PATH = ROOT_PATH + "rap_checkpoints"
SAVE_PATH = ROOT_PATH + 'save_files/'


# Divide corpus
# TODO: optimize split

def split_corpus():
  with open(ROOT_PATH + 'corpus_cleaned_lyrics.txt', 'r') as f:
    data_cleaned = f.readlines()
  # DIvide split in order ; not in random

  train, test = train_test_split(data_cleaned, test_size=0.2)
  train, valid = train_test_split(train, test_size=0.2, random_state=42)

  with open(SAVE_PATH + 'train.txt', 'w') as train_file:
    with open(SAVE_PATH + 'valid.txt', 'w') as valid_file:
      with open(SAVE_PATH + 'test.txt', 'w') as test_file:
        train_file.write(''.join(train))
        valid_file.write(''.join(valid))
        test_file.write(''.join(test))

class Dictionary(object):
  # To start in 0 or 1?

    def __init__(self):
        self.word2idx = {}  # if word2idx["hello"] == 42 ...
        self.idx2word = []  # ... then idx2word[42] == "hello"

    def add_word(self, word):
        """
        This function should check if the word is in word2idx; if it
        is not, it should add it with the first available index
        """
        if word not in self.word2idx:
          self.word2idx[word] = len(self.idx2word) + 1 # It starts in 1
          self.idx2word.append(word)
          return self.word2idx[word]
        else:
          return None

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
  # Dictionnary to token texts train, valid, test
    def __init__(self, path):

    # What is sos token ?

        # We create an object Dictionary associated to Corpus
        self.dictionary = Dictionary()
        self.eos_token = '<eos>'  # end of senetence (line) token
        self.sos_token = '<sos>'
        # We go through all files, adding all words to the dictionary
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))


    def tokenize(self, path):
        """
            Tokenizes a text file, knowing the dictionary, in order to
            tranform it into a list of indices.
            The str.split() function might be useful.
        """
        # Add words to the dictionary
        # Do not forget to add the sos and eos tokens too

        assert os.path.exists(path)
        with open(path, 'r') as f:
          lines = f.readlines()

        for line in lines:
          for word in line.split():
            self.dictionary.add_word(word)

        self.dictionary.add_word(self.eos_token)
        self.dictionary.add_word(self.sos_token)

        # Once done, effectively tokenize by adding the tokens to a vector.
        # We want the `ids` vector to be an int64 torch tensor containing all
        # tokens in the order of the file.
        # Lines should all end with the sos token.

        # TODO
        ids = []

        for line in lines:
          for word in line.split():
            ids.append(self.dictionary.word2idx[word])
          ids.append(self.dictionary.word2idx[self.sos_token])

        ids = np.int64(ids)
        return ids

def batchify(data, bsz):
  # Optimize GPU usage
    """
        Three steps:
        1. Work out how cleanly we can divide the dataset into bsz parts.
        2. Trim off any extra elements that wouldn't cleanly fit (remainders).
        3. Evenly divide the data across the bsz batches.
        Note: You might need to use `.contiguous()` at the end because PyTorch can be somewhat strict about memory usage.
    """
    # Step 1: Calculate the required batch size
    required_bsz = len(data) // (len(data) // bsz)  # Find largest divisor

    # Step 2: Trim the dataset to fit the required batch size
    trimmed_dataset_size = required_bsz * (len(data) // required_bsz)
    trimmed_dataset = data[:trimmed_dataset_size]
    if type(data) == torch.Tensor:
      batches = [trimmed_dataset[i:i + required_bsz * (len(data) // bsz):len(data) // required_bsz].numpy() for i in range(0, len(data) // required_bsz)]
    else:
       batches = [trimmed_dataset[i:i + required_bsz * (len(data) // bsz):len(data) // required_bsz] for i in range(0, len(data) // required_bsz)]

    data_tensor = torch.tensor(batches)

    return data_tensor.to(device)

def get_batch(source, i, seq_len):
    """
        Should return (data, target) where data would be the first variable of
        the example above, and target the second variable.

        - source is the source data;
        - i is the position of the current sequence;
        - seq_len is the sequence length;

        Three steps:
        1. Deal with the possibility that there's not enough data left for a full sequence
        2. Take the input data
        3. Shift by one for the target data
    """
    if i + seq_len > len(source):
      raise
    else:
      data = source[i:i + seq_len]
      target = source[i + 1:i + seq_len + 1]


    return data, target

class LSTMModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.2, initrange=0.1):
        """
            ntoken: length of the dictionary,
            ninp: dimension of the input,
            nhid: dimension of the hidden layers,
            nlayers: number of layers,
            dropout: regularization parameter
            initrange: range for weight initialization
        """
        super().__init__()
        self.ntoken = ntoken
        self.nhid = nhid
        self.nlayers = nlayers
        self.initrange = initrange
        # Create a dropout object to use on layers for regularization
        self.drop = nn.Dropout(dropout)
        # Create an encoder - which is an embedding layer
        self.encoder = nn.Embedding(ntoken, ninp)
        # Create the LSTM layers - find out how to stack them !
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        # Create what we call the decoder: a linear transformation to map the hidden state into scores for all words in the vocabulary
        # (Note that the softmax application function will be applied out of the model)
        self.decoder = nn.Linear(nhid, ntoken)

        # Initialize non-recurrent weights
        self.init_weights()

    def init_weights(self):
        # Initialize the encoder and decoder weights with the uniform distribution,
        # between -self.initrange and +self.initrange, and the decoder bias with zeros
        # https://pytorch.org/docs/stable/nn.init.html?highlight=init
        # - the methods uniform_() and zeros_() might help
        # - self.encoder has a .weight attribute
        # - self.decoder has .weight and .bias attributes

        nn.init.uniform_(self.encoder.weight, -self.initrange, self.initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -self.initrange, self.initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

    def forward(self, input, hidden1):

        # Process the input with the encoder, then dropout
        emb = self.drop(self.encoder(input))

        # Apply the LSTMs
        output, hidden2 = self.rnn(emb, hidden1)

        # Decode into scores
        output = self.drop(output)
        decoded = self.decoder(output)

        return decoded, hidden2

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(BATCH_SIZE)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, SEQ_LEN)):
        try:
          data, targets = get_batch(train_data, i, SEQ_LEN)
        except:
          break

        if optim is not None:
            optim.zero_grad()
        hidden = repackage_hidden(hidden)

        output, hidden = model(data, hidden)

        loss = criterion(output.view(-1, N_TOKENS), targets.view(-1))

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if CLIP is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

        if optim is None:
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)
        else:
            optim.step()

        total_loss += loss.item()

        if batch % LOG_INTERVAL == 0 and batch > 0:
            cur_loss = total_loss / LOG_INTERVAL
            elapsed = time.time() - start_time
            if optim is None:
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // SEQ_LEN, lr,
                    elapsed * 1000 / LOG_INTERVAL, cur_loss, math.exp(cur_loss)))
            else:
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // SEQ_LEN, scheduler.get_last_lr()[-1],
                    elapsed * 1000 / LOG_INTERVAL, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(source):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(EVAL_BATCH_SIZE)
    with torch.no_grad():
        for i in range(0, source.size(0) - 1, SEQ_LEN):
          try:
            data, targets = get_batch(source, i, SEQ_LEN)
          except:
            break
          output, hidden = model(data, hidden)
          hidden = repackage_hidden(hidden)
          total_loss += len(data) * criterion(output.view(-1, N_TOKENS), targets.view(-1)).item()
    return total_loss / (len(source) - 1)

def generate(source, n_words, temperature=1, topk=10):
    """
        n_words: number of words to generate
        fout: optional output file
    """
    vocab_to_int = corpus.dictionary.word2idx
    int_to_vocab = corpus.dictionary.idx2word
    model.eval()
    softmax = nn.Softmax(dim=-1)
    source = source.split()
    hidden = model.init_hidden(1)
    for v in hidden:
        v = v.to(device)
    for w in source:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, hidden = model(ix, hidden)
    output = output / temperature # Here there is only last output of the for
    # To change output and idx_max (ix)
    # To input whole list instead of last item of output

    if topk > 0:
        probas = softmax(torch.topk(softmax(output[0]), topk).values[0]).cpu().detach().numpy()
        indices = torch.topk(softmax(output[0]), topk).indices[0].cpu()
        idx_max = np.random.choice(indices, 1, p=probas)[0]
    else:
        idx_max = torch.argmax(softmax(output[0]))
    words = []
    words.append(int_to_vocab[idx_max])
    for i in range(1, n_words):
        ix = torch.tensor([[idx_max]]).to(device)
        output, hidden = model(ix, hidden)
        output = output / temperature
        if topk > 0:
            probas = softmax(torch.topk(softmax(output[0]), topk).values[0]).cpu().detach().numpy()
            indices = torch.topk(softmax(output[0]), topk).indices[0].cpu()
            idx_max = np.random.choice(indices, 1, p=probas)[0]
        else:
            idx_max = torch.argmax(softmax(output[0]))
        words.append(int_to_vocab[idx_max])
    text = " ".join(words)
    text = text.replace("<eos>", "\n")
    # pp.pprint(text)
    return words



# use CUDA if on a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # corpus c'est le text tokenisé, le corpus doit avoir un dictionnaire
    BATCH_SIZE = 32  # you can choose other values
    EVAL_BATCH_SIZE = 16  # you can choose other values

    split_corpus()

    corpus = Corpus(SAVE_PATH)  # TODO, create the corpus

    train_data = batchify(corpus.train, BATCH_SIZE)
    val_data = batchify(corpus.valid, EVAL_BATCH_SIZE)
    test_data = batchify(corpus.test, EVAL_BATCH_SIZE)

    SEED = 42  # you can choose other values
    torch.manual_seed(SEED) # set the random seed manually for reproducibility.

    EMBEDDING_SIZE = 512  # you can choose other values
    HIDDEN_SIZE = 1024  # you can choose other values
    N_LAYERS = 2  # you can choose other values
    DROPOUT = 0.2  # you can choose other values
    criterion = nn.CrossEntropyLoss()  # maps the output of a Linear layer to a probability distribution
                                    # see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    N_TOKENS = len(corpus.dictionary)  # length of the dictionary
    model = LSTMModel(N_TOKENS, EMBEDDING_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT).to(device)  # initialize the LSTM model

    # You do not need to thoroughly understand the contents of this cell.
    # However, some resources are available in the comments if you are curious.
    # The values are not necessarily optimal, by the way;
    # you could try to tune them later if you have the time.

    LR = 10 # https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10
    OPTIMIZER = 'sgd'  # https://ruder.io/optimizing-gradient-descent/
    WDECAY = 0  # https://d2l.ai/chapter_multilayer-perceptrons/weight-decay.html
    CLIP = 0.25  # https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
    if OPTIMIZER == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WDECAY)
    elif OPTIMIZER == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WDECAY)
    else:
        optim = None

    if OPTIMIZER in ['sgd', 'adam']:
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.7)
    else:
        scheduler = None

    # Other global parameters
    EPOCHS = 10  # number of rounds of training, you can choose other values
    SEQ_LEN = 30  # length of the sequences, you can choose other values
    LOG_INTERVAL = 100  # for logging purposes, you can choose other values

    MODEL_NAME = 'model10'
    SAVE_PATH_MODEL = os.path.join(CHECKPOINT_PATH, '{}.pt'.format(MODEL_NAME))
    LOAD_EXISTING_MODEL = True  # Change to False if you want to start from scratch
                                # ⚠ If False, any existing model with the same name
                                # will be overwritten ⚠

    if LOAD_EXISTING_MODEL:
        try:
            with open(SAVE_PATH_MODEL, 'rb') as f:
                model = torch.load(f)
                # after load the rnn params are not a continuous chunk of memory
                # this makes them a continuous chunk, and will speed up forward pass
                model.rnn.flatten_parameters()
            print("Successfully loaded model from {}".format(SAVE_PATH_MODEL))
        except:
            pass

    best_val_loss = None

    # Loop over epochs.
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        lr = LR
        for epoch in range(1, EPOCHS+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(val_data)
            if scheduler is not None:
                scheduler.step()
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(SAVE_PATH_MODEL, 'wb') as f:
                    torch.save(model, f)
                print("Successfully saved model at {}".format(SAVE_PATH_MODEL))
                best_val_loss = val_loss
            else:
                if scheduler is None:
                    lr /= 4.0

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

        # Load the best saved model.
    with open(SAVE_PATH_MODEL, 'rb') as f:
        model = torch.load(f, map_location=torch.device('cpu'))
        # after load the rn£n params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(val_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    # Generate some text

    source = 'TO START'

    first_word = generate(source, n_words=1, temperature=1, topk=10)[0]
    phrase = []
    phrase.append(first_word)
    for i in range(10):
        word = generate(phrase[i], n_words=1, temperature=1, topk=10)[0]
        phrase.append(word)
    print(source + ' ' + ' '.join(phrase))