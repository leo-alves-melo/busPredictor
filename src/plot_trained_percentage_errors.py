#Encoding: UTF-8
import sys
import pandas as pd
from lib.classes import *
from lib.data_filter import *
from lib.data_processor import *
from lib.data_normalization import *
from lib.training_path import *
import matplotlib.pyplot as plt
from joblib import dump, load
import json

with open('data/corrects_and_errors_rf.json') as arq:
    correctness = json.load(arq)

max_lenght = 100
percentage_1_errors = [0.0]*(max_lenght+1)
percentage_2_errors = [0.0]*(max_lenght+1)
percentage_3_errors = [0.0]*(max_lenght+1)
percentage_4_errors = [0.0]*(max_lenght+1)

percentage_1_correct = [0.0]*(max_lenght+1)
percentage_2_correct = [0.0]*(max_lenght+1)
percentage_3_correct = [0.0]*(max_lenght+1)
percentage_4_correct = [0.0]*(max_lenght+1)


for lenght in range(max_lenght):
    
    try:
        percentage_1_errors[lenght+1] = 100 * float(correctness['error_rf_1_' + str(lenght+1)])/float(correctness['total_1_' + str(lenght+1)])
        percentage_2_errors[lenght+1] = 100 * float(correctness['error_rf_2_' + str(lenght+1)])/float(correctness['total_2_' + str(lenght+1)])
        percentage_3_errors[lenght+1] = 100 * float(correctness['error_rf_3_' + str(lenght+1)])/float(correctness['total_3_' + str(lenght+1)])
        percentage_4_errors[lenght+1] = 100 * float(correctness['error_rf_4_' + str(lenght+1)])/float(correctness['total_4_' + str(lenght+1)])

        percentage_1_correct[lenght+1] = 100 * float(correctness['correct_rf_1_' + str(lenght+1)])/float(correctness['total_1_' + str(lenght+1)])
        percentage_2_correct[lenght+1] = 100 * float(correctness['correct_rf_2_' + str(lenght+1)])/float(correctness['total_2_' + str(lenght+1)])
        percentage_3_correct[lenght+1] = 100 * float(correctness['correct_rf_3_' + str(lenght+1)])/float(correctness['total_3_' + str(lenght+1)])
        percentage_4_correct[lenght+1] = 100 * float(correctness['correct_rf_4_' + str(lenght+1)])/float(correctness['total_4_' + str(lenght+1)])
    except:
        pass

print('plotando...')

plt.plot(percentage_4_errors, marker='', color='green', label=u"Falsos positivos")
plt.plot(percentage_4_correct, marker='', markerfacecolor='blue', label="Verdadeiros positivos")
plt.legend(loc='best')
plt.xlabel(u'Porcentagem de completude do caminho')
plt.ylabel(u'Porcentagem de acerto ou erro')
plt.title(u'Linha 4: Falsos e Verdadeiros Positivos x Porcentagem de completude do caminho')
plt.grid(True)
plt.show()
plt.clf()