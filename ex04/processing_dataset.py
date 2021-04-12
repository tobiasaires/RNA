import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def concat_all_lines_in_column(pd_dataset, column_name):
    result_concat = []

    for line in range(pd_dataset[column_name].size): #pd_dataset[column_name].size
        result_concat = np.concatenate(
        (result_concat, str(pd_dataset[column_name][line]).lower().split()),
            axis=0
        )

    return result_concat

def convert_string_text(array):
    result_array = []

    for text in array:
        result_array.append(str(text).lower())
    
    return result_array

def remove_and_insert(pd_dataset, column_name, value):
    result_remove = []

    for line in range(pd_dataset[column_name].size): #pd_dataset[column_name].size
        element = str(pd_dataset[column_name][line])

        if element.isnumeric():
            result_remove = np.concatenate(
            (result_remove, element.split()),
                axis=0
            )
        else:
            result_remove = np.concatenate(
            (result_remove, [value]),
                axis=0
            )

    return result_remove

# carregando dataset
table = pd.read_csv('chennai_reviews.csv', sep=',')

# criando dicionario (sem repeticoes) da coluna Review_Text
dictionary = np.unique(concat_all_lines_in_column(table, 'Review_Text'))

# guardando o tamanho do dicionario
n_words = len(dictionary)

# guardando a coluna de predicoes 'Sentiment' inserindo '2' nos dados invalidos
all_sentiment = remove_and_insert(table, 'Sentiment', '2')
all_text_review = convert_string_text(table['Review_Text'].tolist())

all_categories = np.unique(all_sentiment).tolist()
n_categories = len(all_categories)

# print(all_text_review)
# exit()

''' word_to_tensor(), text_to_tensor()
    Estas funcoes irao transformar as palavras de um dicionario qualquer em vetores binarios.
    O formato da binarizacao sera um vetor do tamanho do dicionario com seus valores iguais
    a 0, exceto o indice do qual representara a palavra, pois este valor sera 1.
    Ex:
        dicionario = ['Estou', 'em', 'casa.']
    Sendo assim:
        'Estou' = [1, 0, 0]
        'em' = [0, 1, 0]
        'casa.' = [0, 0, 1]
    Logo:
        'Estou em casa.' = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
'''
''' funcao que monta o vetor binario dado uma palavra '''
def word_to_tensor(word):
    tensor = torch.zeros(1, n_words)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

''' funcao que monta uma matriz binaria dado um texto '''
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_words)
    for li, word in enumerate(line):
        tensor[li][0][word_to_index(word)] = 1
    return tensor

    
''' funcao encontra o index da palvra '''
def word_to_index(word):
    itemindex, = np.where(dictionary==word)
    # print(itemindex[0])
    if(itemindex.size > 0):
        return itemindex[0]
    else:
        return 0
'''
print(dictionary)
print(word_to_tensor('its'))
print(text_to_tensor('its really nice place'))
'''

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_words, n_hidden, n_categories)

''' Teste com palavra '''
# input = wordToTensor('before')
# hidden = torch.zeros(1, n_hidden)
# output, next_hidden = rnn(input, hidden)
# print(output)

''' Teste com texto '''
# input = line_to_tensor('its nice')
# hidden = torch.zeros(1, n_hidden)
# output, next_hidden = rnn(input[0], hidden)
# print(output)

''' 
    funcao criada para "traduzir" a saida da rede (output) que e um tensor com
    as probabilidades de cada categoria 
'''
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

'''
    funcao criada para randomizar uma entrada inicial para o treinamento
'''
def randomTrainingExample():
    n_random = random.randint(0, len(all_text_review) - 1) #len(all_text_review)

    category = all_sentiment[n_random]
    line = all_text_review[n_random]

    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)

    return category, line, category_tensor, line_tensor

# for i in range(10):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     print('category =', category, '/ caregory_tensor =', category_tensor)


'''
    aqui contem as variaveis necessarias e a funcao para treinamento dos dados
'''
criterion = nn.NLLLoss()

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

n_iters = 1000
print_every = 50
plot_every = 10

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

plt.figure()
plt.plot(all_losses)
plt.show()

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 100

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()