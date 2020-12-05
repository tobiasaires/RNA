import math

# funções de ativação -----------------------------------------------------
def sign(alfa):
    if alfa <= 0:
        return -1
    else:
        return 1

def logistic(alfa):
    return 1 / 1 + math.exp(-alfa)

def tanh(alfa):
    return (math.exp(2 * alfa) - 1) / (math.exp(2 * alfa) + 1)

def relu(alfa):
    if alfa <= 0:
        return 0
    else:
        return alfa
# --------------------------------------------------------------------------

# função que calcula o produto interno de dois vetores ---------------------
def inner_product(v1, v2):
    result = 0
    for i in range(len(v1)):
        result += float(v1[i]) * float(v2[i])
    return result
# --------------------------------------------------------------------------

# função que multiplica um elemento por um vetor ---------------------------
def product(e, v):
    result = []
    for i in range(len(v)):
        x = float(e) * float(v[i])
        result.append(x)
    return result
# --------------------------------------------------------------------------

# função que soma dois vetores ---------------------------------------------
def sum(v1, v2):
    result = []
    for i in range(len(v1)):
        x = float(v1[i]) + float(v2[i])
        result.append(x)
    return result
# --------------------------------------------------------------------------

# função que calcula a porcentagem de similaridade entredois vetores -------
def score(v1, v2):
    hits = 0
    for i in range(len(v1)):
        if(float(v1[i]) == float(v2[i])):
            hits += 1
    return hits/len(v1)
# --------------------------------------------------------------------------

# coletando os dados de entada X e y ---------------------------------------
X = []
y = []

arq_address = r'data_or.dat'
f = open(arq_address,"r")
row = f.readline().replace('\n','')
while row:
    columns = row.split(" ")
    # guardo todas as colunas, menos a última (predição)
    X.append(columns[:len(columns) - 1])
    # guardo a última coluna (predição)
    y.append(columns[-1])
    row = f.readline().replace('\n', '')
f.close()
# --------------------------------------------------------------------------

# pesos de conexão ---------------------------------------------------------
W = [0,0] # todo (automatizar a criação, pois deve ter o mesmo tamanho de X 
b = W[0]
# --------------------------------------------------------------------------

# efetuando o treinamento do Perceptron ------------------------------------
# predições treinadas
y_train = []
# número de iterações t do treinamento
T = 100
for t in range(T):
    # guardara as predições em treinamento
    y_training = []

    # percorre todas as predições (vetor y)
    for n in range(len(y)):
        # calculando a predição do perceptron aplicando a função de ativação
        yn = sign(inner_product(W, X[n]))

        # averiguando se houve erro de classificação para aplicar a correção
        if y[n] != yn:
            W = sum(W, product(y[n], X[n]))

        y_training.append(yn)
    
    # atualizando as predições
    y_train = y_training
# --------------------------------------------------------------------------
print(W)
print(score(y, y_training))