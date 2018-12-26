#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:41:33 2018

@author: linweili
"""
import pandas as pd
import sys
import evalResultHelper as helper
import math


if len(sys.argv) <= 1:
    lang = 'EN'
else:
    lang = sys.argv[1]
    
# read training data
x=[]
y=[]
word_dict = []
label_dict = []

with open(lang+'/train','r') as f:
    x_sentence = []
    y_sentence = []
    for line in f:
        if line=='\n':
            x.append(x_sentence)
            y.append(y_sentence)
            x_sentence=[]
            y_sentence=[]
        else:
            raw = line.strip('\n')
            raw = raw.split()
            word = raw[0]
            label = raw[1]
            x_sentence.append(word)
            y_sentence.append(label)
            if word not in word_dict:
                word_dict.append(word)
            if label not in label_dict:
                label_dict.append(label)

def read_test(filename):
    test_x = []
    with open(filename,'r') as f:
        x_sentence = []

        for line in f:
            if line=='\n':
                test_x.append(x_sentence)
                x_sentence=[]

            else:
                word = line.strip('\n')
                x_sentence.append(word)
    return test_x
                
                
def output_predict(out_x,out_y, filename):
    f_out = open(filename, "w")
    for sentence_index in range(len(out_x)):
        for word_index in range(len(out_x[sentence_index])):
            word = out_x[sentence_index][word_index]
            label = out_y[sentence_index][word_index]
            f_out.write(word+ ' ' + label + '\n')
        f_out.write('\n')
    f_out.close()


# train emission, and transition
emission = pd.DataFrame(0,index=label_dict, columns=word_dict+['#!UNK!#'])
transition = pd.DataFrame(0,index = (['START']+label_dict), columns=(label_dict+['END']))

emission = emission.to_dict('index')
transition = transition.to_dict('index')

for sentence_index in range(len(x)):
    sentence = x[sentence_index]
        #emission
    for word_index in range(len(sentence)):
        word = sentence[word_index]
        label = y[sentence_index][word_index]
        emission[label][word]+=1
    
    #transition
    transition['START'][y[sentence_index][0]] +=1
    
    transition[y[sentence_index][-1]]['END']+=1
    
    for word_index in range(len(sentence)-1):
        label1 = y[sentence_index][word_index]
        label2 = y[sentence_index][word_index+1]
        transition[label1][label2]+=1
for label_key in emission.keys():
    emission[label_key]['#!UNK!#'] = 1

for dictionary in emission.values():
    total = sum(dictionary.values())
    for word_key in dictionary.keys():
        dictionary[word_key] /= total
        
for dictionary in transition.values():
    total = sum(dictionary.values())
    for label_key in dictionary.keys():
        dictionary[label_key] /= total

predict_y = []
test_x = read_test(lang+'/dev.in')

def find_max_key1(dictionary, word):
    max_value = 0
    result = None
    for key in dictionary.keys():
        if dictionary[key][word] > max_value:
            max_value = dictionary[key][word]
            result = key
    return result


for sentence_index in range(len(test_x)):
    predict_y_sentence = []
    for word_index in range(len(test_x[sentence_index])):
        word = test_x[sentence_index][word_index]
        clean_word = word
        if word not in word_dict:
                clean_word = '#!UNK!#'
        predicted_label = find_max_key1(emission, clean_word)
        predict_y_sentence.append(predicted_label)
        
    predict_y.append(predict_y_sentence)
    



output_predict(test_x,predict_y,lang+'/dev.p2.out')

gold_file = open(lang+'/dev.out')
prediction_file = open(lang+'/dev.p2.out')

observed = helper.get_observed(gold_file)

predicted = helper.get_predicted(prediction_file)

#Compare
print('emission:')
helper.compare_observed_to_predicted(observed, predicted, discardInstance=[])


#transition
def find_max_key2(dictionary):
    max_value = 0
    result = None
    for key in dictionary.keys():
        if dictionary[key] > max_value:
            max_value = dictionary[key]
            result = key
    return result

predict_y = []
for sentence_index in range(len(test_x)):
    predict_y_sentence = []
    pre_label='START'
    for word_index in range(len(test_x[sentence_index])):
        label = find_max_key2(transition[pre_label])
        predict_y_sentence.append(label)
        pre_label = label
    predict_y.append(predict_y_sentence)

output_predict(test_x,predict_y,lang+'/dev.p3_transition.out')

try:
    gold_file = open(lang+'/dev.out')
    prediction_file = open(lang+'/dev.p3_transition.out')
    observed = helper.get_observed(gold_file)
    predicted = helper.get_predicted(prediction_file)
except:
    print("transition is 0")




#viterbi
test_x = read_test(lang+'/dev.in')
predict_y = []

def convert_unknown(sentence):
    result=[]
    for word in sentence:
        if word not in word_dict:
            result.append('#!UNK!#')
        else:
            result.append(word)
    return result

def log10(value):
    if value==0:
        return -10000
    else:
        return math.log10(value)
    
    
def viterbi(x_sentence):
    
    predict_y_sentence = []
    matrix = [[0 for x in range(len(x_sentence))] for y in range(len(label_dict))]
    backward_linked_table = [[0 for x in range(len(x_sentence))] for y in range(len(label_dict))]
    
    for i in range(len(label_dict)):
        matrix[i][0] = log10(transition['START'][label_dict[i]])+log10(emission[label_dict[i]][x_sentence[0]])                                                                  
    for j in range(1,len(x_sentence)):
        for i in range(len(label_dict)):  
            max_value= 0 - math.inf
            for k in range(len(label_dict)):
                value = matrix[k][j-1]+log10(transition[label_dict[k]][label_dict[i]])+log10(emission[label_dict[i]][x_sentence[j]])
                if value>max_value:
                    max_pre_label_index = k
                    max_value = value
            backward_linked_table[i][j] = max_pre_label_index
            matrix[i][j] = max_value
            
    end_max = 0-math.inf
    max_pre_label = None
    for i in range(len(label_dict)):
        value = matrix[i][len(x_sentence)-1]+log10(transition[label_dict[i]]['END'])
        if value>end_max:
            end_max=value
            max_pre_label_index = i
    
    predict_y_sentence.insert(0,label_dict[max_pre_label_index])
    for j in reversed(range(1,len(x_sentence))):
        pre_label = label_dict[backward_linked_table[max_pre_label_index][j]]
        max_pre_label_index = backward_linked_table[max_pre_label_index][j]
        predict_y_sentence.insert(0,pre_label)
        
    return predict_y_sentence                                                                                       
    

                                                                                              
for sentence_index in range(len(test_x)):
    clean_sentence = convert_unknown(test_x[sentence_index])
    predict_y_sentence = viterbi(clean_sentence)
    predict_y.append(predict_y_sentence)


output_predict(test_x,predict_y,lang+'/dev.p3.out')

gold_file = open(lang+'/dev.out')
prediction_file = open(lang+'/dev.p3.out')
observed = helper.get_observed(gold_file)
predicted = helper.get_predicted(prediction_file)

print('\nViterbi:')
helper.compare_observed_to_predicted(observed, predicted, discardInstance=[])




# second order
import pandas as pd

emission = pd.DataFrame(0,index=label_dict, columns=word_dict+['#!UNK!#'])
emission = emission.to_dict('index')

transition={}
for label1 in ['START']+label_dict:
    first_level = {}
    if label1=='START':
        for label2 in ['START']+label_dict+['END']:
            if label2=='END':
                first_level[label2] = {'END':0}
            else:
                second_level = {}
                for label3 in label_dict+['END']:
                    second_level[label3] = 0
                first_level[label2] = second_level
    else:
        for label2 in label_dict+['END']:
            if label2=='END':
                first_level[label2] = {'END':0}
            else:
                second_level = {}
                for label3 in label_dict+['END']:
                    second_level[label3] = 0
                first_level[label2] = second_level
    transition[label1]=first_level
    
    
for sentence_index in range(len(x)):
    sentence = x[sentence_index]
    
    #emission
    for word_index in range(len(sentence)):
        word = sentence[word_index]
        label = y[sentence_index][word_index]
        emission[label][word]+=1
    
    #transition
    transition['START']['START'][y[sentence_index][0]] +=1
    transition[y[sentence_index][-1]]['END']['END']+=1
    if len(sentence) >=2:
        transition['START'][y[sentence_index][0]][y[sentence_index][1]] +=1
        transition[y[sentence_index][-2]][y[sentence_index][-1]]['END']+=1
        
    for word_index in range(len(sentence)-2):
        label1 = y[sentence_index][word_index]
        label2 = y[sentence_index][word_index+1]
        label3 = y[sentence_index][word_index+2]
        transition[label1][label2][label3]+=1
            
for label_key in emission.keys():
    emission[label_key]['#!UNK!#'] = 1

for dictionary in emission.values():
    total = sum(dictionary.values())
    for word_key in dictionary.keys():
        dictionary[word_key] /= total
        
for dictionary in transition.values():
    for dictionary2 in dictionary.values():
        total = sum(dictionary2.values())
        for label_key in dictionary2.keys():
            if total!=0:
                dictionary2[label_key] /= total


test_x = read_test(lang+'/dev.in')
predict_y = []

def convert_unknown(sentence):
    result=[]
    for word in sentence:
        if word not in word_dict:
            result.append('#!UNK!#')
        else:
            result.append(word)
    return result

def log10(value):
    if value==0:
        return -10000
    else:
        return math.log10(value)
    
    
def viterbi(x_sentence):
    
    predict_y_sentence = []
    matrix = [[[0 for x in range(len(x_sentence))] for y in range(len(label_dict))] for y in range(len(label_dict))]
    backward_linked_table = [[[0 for x in range(len(x_sentence))] for y in range(len(label_dict))] for y in range(len(label_dict))]
    
    for i in range(len(label_dict)):
        for ii in range(len(label_dict)):
            matrix[i][ii][1] = log10(transition['START']['START'][label_dict[i]])+log10(emission[label_dict[i]][x_sentence[0]]) + log10(transition['START'][label_dict[i]][label_dict[ii]]) + log10(emission[label_dict[ii]][x_sentence[1]])                                                                  
    for j in range(2,len(x_sentence)):
        for i in range(len(label_dict)): 
            for ii in range(len(label_dict)):
                max_value= 0 - math.inf
                for k in range(len(label_dict)):
                    value = matrix[k][i][j-1]+log10(transition[label_dict[k]][label_dict[i]][label_dict[ii]])+log10(emission[label_dict[ii]][x_sentence[j]])
                    if value>max_value:
                        max_pre_label_index = k
                        max_value = value
                backward_linked_table[i][ii][j] = max_pre_label_index
                matrix[i][ii][j] = max_value
            
    end_max = 0-math.inf
    max_pre_label = None
    for i in range(len(label_dict)):
        for ii in range(len(label_dict)):
            value = matrix[i][ii][len(x_sentence)-1]+log10(transition[label_dict[i]][label_dict[ii]]['END'])
            if value>end_max:
                end_max=value
                max_pre_label_index = ii
                max_pre2_label_index = i
    
    predict_y_sentence.insert(0,label_dict[max_pre_label_index])
    predict_y_sentence.insert(0,label_dict[max_pre2_label_index])
    for j in reversed(range(2,len(x_sentence))):
        temp = max_pre2_label_index
        pre_label = label_dict[backward_linked_table[max_pre2_label_index][max_pre_label_index][j]]
        max_pre2_label_index = backward_linked_table[max_pre2_label_index][max_pre_label_index][j]
        max_pre_label_index = temp
        predict_y_sentence.insert(0,pre_label)
        
    return predict_y_sentence                                                                                       
    

                                                                                              
for sentence_index in range(len(test_x)):
    clean_sentence = convert_unknown(test_x[sentence_index])
    predict_y_sentence = viterbi(clean_sentence)
    predict_y.append(predict_y_sentence)
    

output_predict(test_x,predict_y,lang+'/dev.p4.out')

import evalResultHelper as helper

gold_file = open(lang+'/dev.out')
prediction_file = open(lang+'/dev.p4.out')
observed = helper.get_observed(gold_file)
predicted = helper.get_predicted(prediction_file)
print('\nSecond Order:')
helper.compare_observed_to_predicted(observed, predicted, discardInstance=[])
