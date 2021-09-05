import json

def calculateArea(percent):
    summation = 0
    for value in percent:
        summation += value

    return summation


with open('../data/correctness_percentage_all_ensembled.json') as arq:
    correctness = json.load(arq)
with open('../data/correctness_bayes.json') as arq:
    correctness_bayes = json.load(arq)
with open('../data/correct_times_series.json') as arq:
    correctness_time_series = json.load(arq)
with open('../data/correctness_riobus_completed.json') as arq:
    correctness_rio_bus = json.load(arq)

max_lenght = 100
percentage_svm = [0.0]*(max_lenght)
percentage_nn = [0.0]*(max_lenght)
percentage_dump = [0.0]*(max_lenght)
percentage_rf = [0.0]*(max_lenght)
percentage_ensembled_rf = [0.0]*(max_lenght)
percentage_bayes = [0.0]*(max_lenght)
percentage_time_series = [0.0]*(max_lenght)
percentage_riobus = [0.0]*(max_lenght)
for lenght in range(max_lenght):
    percentage_svm[lenght] = 100 * float(correctness['correct_svm_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
    percentage_dump[lenght] = 100 * float(correctness['correct_dump_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
    percentage_rf[lenght] = 100 * float(correctness['correct_rf_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
    percentage_nn[lenght] = 100 * float(correctness['correct_conv_nn_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
    percentage_bayes[lenght] = 100 * float(correctness_bayes['correct_bayes_' + str(lenght+100)])/float(correctness_bayes['total_' + str(lenght+100)])
    percentage_ensembled_rf[lenght] = 100 * float(correctness['correct_ensembled_rf_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
    percentage_time_series[lenght] = 100 * float(       correctness_time_series['correct_rf_' + str(lenght+1) + "_1"] + correctness_time_series['correct_rf_' + str(lenght+1) + "_2"] + correctness_time_series['correct_rf_' + str(lenght+1) + "_3"] + correctness_time_series['correct_rf_' + str(lenght+1) + "_4"] )/float( correctness_time_series['total_' + str(lenght+1) + "_1"] + correctness_time_series['total_' + str(lenght+1) + "_2"] + correctness_time_series['total_' + str(lenght+1) + "_3"] + correctness_time_series['total_' + str(lenght+1) + "_4"]  )
    percentage_riobus[lenght] = 100 * float(correctness_rio_bus['correct_conv_nn_' + str(lenght+1)])/float(correctness_rio_bus['total_' + str(lenght+1)])

print('svm: ', calculateArea(percentage_svm))
print('dump: ', calculateArea(percentage_dump))
print('percentage_rf: ', calculateArea(percentage_rf))
print('percentage_nn: ', calculateArea(percentage_nn))
print('percentage_bayes: ', calculateArea(percentage_bayes))
print('percentage_ensembled_rf: ', calculateArea(percentage_ensembled_rf))
print('percentage_time_series: ', calculateArea(percentage_time_series))
print('percentage_riobus: ', calculateArea(percentage_riobus))
