{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Equipe : Gabriel Monteiro e Tobias"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Questão 2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Explicar o problema associado ao dataset escolhido;\n",
    "`O dataset escolhido faz referência ao dataset MNIST, porém de uma maneira mais simplificada e reduzida. No original elas são representas 28x28, neste toy_dataset ele é somente 8x8`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.datasets import load_digits\n",
    "from sklearn.model_selection import train_test_split\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "digits = load_digits()\n",
    "\n",
    "X = digits.data\n",
    "y = digits.target\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 432x288 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.gray() \n",
    "plt.matshow(digits.images[0]) \n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/raul/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/_multilayer_perceptron.py:571: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10) reached and the optimization hasn't converged yet.\n",
      "  % self.max_iter, ConvergenceWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,\n",
       "              beta_2=0.999, early_stopping=False, epsilon=1e-08,\n",
       "              hidden_layer_sizes=10, learning_rate='constant',\n",
       "              learning_rate_init=0.3, max_fun=15000, max_iter=10, momentum=0.9,\n",
       "              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,\n",
       "              random_state=None, shuffle=True, solver='adam', tol=0.0001,\n",
       "              validation_fraction=0.1, verbose=False, warm_start=False)"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf_1 = MLPClassifier(hidden_layer_sizes=10, learning_rate_init=0.3, max_iter=10)\n",
    "\n",
    "clf_1.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,\n",
       "              beta_2=0.999, early_stopping=False, epsilon=1e-08,\n",
       "              hidden_layer_sizes=100, learning_rate='constant',\n",
       "              learning_rate_init=0.03, max_fun=15000, max_iter=100,\n",
       "              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,\n",
       "              power_t=0.5, random_state=None, shuffle=True, solver='adam',\n",
       "              tol=0.0001, validation_fraction=0.1, verbose=False,\n",
       "              warm_start=False)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf_2 = MLPClassifier(hidden_layer_sizes=100, learning_rate_init=0.03, max_iter=100)\n",
    "\n",
    "clf_2.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,\n",
       "              beta_2=0.999, early_stopping=False, epsilon=1e-08,\n",
       "              hidden_layer_sizes=1000, learning_rate='constant',\n",
       "              learning_rate_init=0.003, max_fun=15000, max_iter=1000,\n",
       "              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,\n",
       "              power_t=0.5, random_state=None, shuffle=True, solver='adam',\n",
       "              tol=0.0001, validation_fraction=0.1, verbose=False,\n",
       "              warm_start=False)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf_3 = MLPClassifier(hidden_layer_sizes=1000, learning_rate_init=0.003, max_iter=1000)\n",
    "\n",
    "clf_3.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Plotar a evolução da função custo (loss) ao longo do treinamento (épocas) para verificar a\n",
    "corretude do algoritmo de treinamento."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "loss_values_digits_1 = clf_1.loss_curve_\n",
    "plt.plot(loss_values_digits_1)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "loss_values_digits_2 = clf_2.loss_curve_\n",
    "plt.plot(loss_values_digits_2)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "loss_values_digits_3 = clf_3.loss_curve_\n",
    "plt.plot(loss_values_digits_3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Reportar taxas de acerto (ou erro) nos conjuntos de treinamento e teste para diferentes\n",
    "configurações da rede neural."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "acc_1 = 0.08444444444444445\n"
     ]
    }
   ],
   "source": [
    "y_predict_1 = clf_1.predict(X_test)\n",
    "acc_1 = accuracy_score(y_test, y_predict_1)\n",
    "print('acc_1 = {}'.format(acc_1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "acc_2 = 0.9711111111111111\n"
     ]
    }
   ],
   "source": [
    "y_predict_2 = clf_2.predict(X_test)\n",
    "acc_2 = accuracy_score(y_test, y_predict_2)\n",
    "print('acc_2 = {}'.format(acc_2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "acc_3 = 0.9844444444444445\n"
     ]
    }
   ],
   "source": [
    "y_predict_3 = clf_3.predict(X_test)\n",
    "acc_3 = accuracy_score(y_test, y_predict_3)\n",
    "print('acc_3 = {}'.format(acc_3))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}