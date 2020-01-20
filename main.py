import torch
from torchtext import data

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

import spacy
spacy_en = spacy.load('en_core_web_sm')

def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


#define the fields:
from torchtext import data
from torchtext import datasets

NAME = data.Field(sequential=True, tokenize=tokenizer, lower=True)
SAYING = data.Field(sequential=True, tokenize=tokenizer, lower=True)
PLACE = data.Field(sequential=True, tokenize=tokenizer, lower=True)

fields = {'name': ('n', NAME), 'location': ('p', PLACE), 'quote': ('s', SAYING)}

train_data, valid_data, test_data = data.TabularDataset.splits(
                                        path = './data',
                                        train = 'train.json',
                                        validation = 'valid.json',
                                        test = 'test.json',
                                        format = 'json',
                                        fields = fields )

# print(vars(train_data[0]))

# create a `Vector` object

import torchtext.vocab as vocab

custom_embeddings = vocab.Vectors(name = './custom_embeddings/embeddings.txt',
                                  cache = './custom_embeddings',
                                  unk_init = torch.Tensor.normal_)
# print(type(custom_embeddings))



MAX_VOCAB_SIZE = 25_000 # MAX_VOCAB_SIZE:25000, <class 'int'>

## Iterators
NAME.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, vectors = custom_embeddings)
SAYING.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, vectors = custom_embeddings)
PLACE.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, vectors = custom_embeddings)

print(NAME.vocab.vectors[NAME.vocab.stoi['good']])

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort=False,  # don't sort test/validation data
    #sort_key=lambda x: x.s,  # sort by s attribute (quote)
    batch_size=BATCH_SIZE,
    device=device)

print('Train:')
for batch in train_iterator:
    print(batch)

print('Valid:')
for batch in valid_iterator:
    print(batch)

print('Test:')
for batch in test_iterator:
    print(batch)


# import our model

from models.CNN import CNN as CNN

INPUT_DIM = len(NAME.vocab)
EMBEDDING_DIM = 20
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = NAME.vocab.stoi[NAME.pad_token]

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


embeddings = NAME.vocab.vectors

model.embedding.weight.data.copy_(embeddings)

# initialize the unknown and padding token embeddings to all zeros.
UNK_IDX = NAME.vocab.stoi[NAME.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

import torch.optim as optim

optimizer = optim.Adam(model.parameters())

#Define our loss function (criterion).

import torch.nn as nn
criterion = nn.BCEWithLogitsLoss()


model = model.to(device)
criterion = criterion.to(device)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


# training function...

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
FREEZE_FOR = 5

best_valid_loss = float('inf')

# freeze embeddings
model.embedding.weight.requires_grad = unfrozen = False

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | Frozen? {not unfrozen}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tutC-model.pt')

    if (epoch + 1) >= FREEZE_FOR:
        # unfreeze embeddings
        model.embedding.weight.requires_grad = unfrozen = True