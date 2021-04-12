# RNA-rnn

## Instalação e Execução
Clone o repositório:
```
git clone git@github.com:daniloaldm/RNA-rnn.git
```
Vá para o diretório:
```
cd RNA-rnn
```
Execute:
```
python3 -m venv .
. bin/activate
pip3 install -r requirements.txt
```
Para executar os arquivos no jupiter:
```
jupyter notebook
```

## Resultados
Foi utilizado a base de dados do arquivo chennai_reviews.csv, uma taxa de treinamento de 0.005 e 1000 iterações.

Abaixo podemos ver alguns resultados obtidos durante a execução do treino em alguns exemplos. O formato dos resultados são: **texto / predição ✓** ou **texto / predição ✗ (predição correta)**.

![Figure_3](https://user-images.githubusercontent.com/51512175/114290613-45464f00-9a57-11eb-8a5f-55e48aaf5660.png)

Abaixo temos um gráfico que representa a perda histórica e que mostra o aprendizado da rede.

![Figure_1](https://user-images.githubusercontent.com/51512175/114290668-a0784180-9a57-11eb-82ba-f12a6af68eac.png)

Para ver o desempenho da rede em diferentes categorias, usamos o método descrito em [pytorch.org](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html), alterando o número de iterações de 10000 para 100. O resultado obtido é mostrado abaixo.

![Figure_2](https://user-images.githubusercontent.com/51512175/114290826-b2a6af80-9a58-11eb-8ed8-e98470d64238.png)