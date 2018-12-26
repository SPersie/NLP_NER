import argparse
from random import shuffle

"""
READ IN TRAINING DATA
"""
def readtrainingdata(training):
    x=[]
    y=[]
    word_dict = []
    label_dict = []

    with open(training,'r') as f:
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
                word = raw[0].lower()
                label = raw[1]
                x_sentence.append(word)
                y_sentence.append(label)
                if word not in word_dict:
                    word_dict.append(word)
                if label not in label_dict:
                    label_dict.append(label)
    return x, y, word_dict, label_dict


"""
FUNCTION TO SHUFFLE THE INPUT DATASET

1. implement shuffle and average
2. shuffle the training data at sentence level
"""
def shuffle_xy(x, y):
    xy = []
    shuffle_x = []
    shuffle_y = []
    for i in range(len(x)):
        sentence = []
        xy.append(sentence)
        for j in range(len(x[i])):
            word_label = []
            xy[i].append(word_label)
            xy[i][j].append(x[i][j])
            xy[i][j].append(y[i][j])

    shuffle(xy)

    for i in range(len(xy)):
        x_sentence = []
        y_sentence = []
        shuffle_x.append(x_sentence)
        shuffle_y.append(y_sentence)
        for j in range(len(xy[i])):
            shuffle_x[i].append(xy[i][j][0])
            shuffle_y[i].append(xy[i][j][1])

    return shuffle_x, shuffle_y


"""
READ IN TESTING DATA
"""
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
                x_sentence.append(word.lower())
    return test_x


"""
CONSTRUCT FEATURE MATRIX

1. transition and emission matrices are constructed respectively
2. set a threshold k to replace word with ##UNK##
"""
def constructfeature(x, word_dict, label_dict, k=1):
    words = {}
    for sentence in x:
        for word in sentence:
            if word not in words:
                words[word] = 1
            else:
                words[word] += 1

    # replace word appeared less than k times with ##UNK##
    for word in word_dict:
        if words[word] < k:
            word_dict.remove(word)
    word_dict.append('UNK')

    # generate transition feature matrix
    tranfeature = {}

    u_labels = label_dict.copy()
    u_labels.append('START')
    v_labels = label_dict.copy()
    v_labels.append('STOP')

    for u_label in u_labels:
        tranfeature[u_label] = {}
        for v_label in v_labels:
            tranfeature[u_label][v_label] = [0.0]

    # generate emission feature matrix
    emitfeature = {}
    for label in label_dict:
        emitfeature[label] = {}
        for word in word_dict:
            emitfeature[label][word] = [0.0]

    return tranfeature, emitfeature


"""
VITERBI

1. improvement: change the calculation of score
"""
def viterbi(sentence, word_dict, label_dict, tranfeature, emitfeature):
    bltable = [{label: [float('-inf'), ''] for label in label_dict} for word in sentence]

    # first transition from START
    for label in label_dict:
        # update label
        bltable[0][label][1] = 'START'
        score = 0.0
        score += tranfeature['START'][label][0]
        # update score
        ## score from transition
        ## score from emission
        if sentence[0] in word_dict:
            score += emitfeature[label][sentence[0]][0]
        else:
            score += emitfeature[label]['UNK'][0]
        bltable[0][label][0] = score

    # calculate score for all nodes
    for i in range(1, len(sentence)):
        for u in label_dict:
            for v in label_dict:
                # score from previsou layer
                score = bltable[i-1][v][0]
                # add transition weight to score
                score += tranfeature[v][u][0]
                # find the maximum score
                if score > bltable[i][u][0]:
                    bltable[i][u] = [score, v]

            # add emission weight to score (independent to transition)
            ## check whether exist
            if sentence[i] in word_dict:
                bltable[i][u][0] += emitfeature[u][sentence[i]][0]
            else:
                bltable[i][u][0] += emitfeature[u]['UNK'][0]

    # last transition to stop
    stop = [float('-inf'), '']
    for label in label_dict:
        score = bltable[-1][label][0] + tranfeature[label]['STOP'][0]

        if score > stop[0]:
            stop = [score, label]

    # back propogation to find labels
    sentence_prediction = [stop[1]]
    for i in reversed(range(len(sentence))):
        if i == 0:
            break
        else:
            sentence_prediction.insert(0, bltable[i][sentence_prediction[0]][1])

    if 'START' in sentence_prediction:
        print(True)
        print(sentence_prediction)

    return sentence_prediction


"""
UPDATE FEATURE MATRICES
"""
def updatefeature(sentence, word_dict, labels, predict_labels, tranfeature, emitfeature):
    labels.insert(0, 'START')
    predict_labels.insert(0, 'START')
    labels.append('STOP')
    predict_labels.append('STOP')
    sentence.insert(0, '')
    sentence.append('')

    for i in range(len(labels)):
        # error driven algo
        if labels[i] != predict_labels[i]:
            # update for transition feature matrix
            tranfeature[labels[i-1]][labels[i]][0] += 1
            tranfeature[labels[i]][labels[i+1]][0] += 1

            tranfeature[predict_labels[i-1]][predict_labels[i]][0] -= 1
            tranfeature[predict_labels[i]][predict_labels[i+1]][0] -= 1

            # update for emission feature matrix
            ## if the word exists in the word_dict
            if sentence[i] in word_dict:
                emitfeature[labels[i]][sentence[i]][0] += 1

                emitfeature[predict_labels[i]][sentence[i]][0] -= 1

            ## if the word does not exist, update UNK
            else:
                emitfeature[labels[i]]['UNK'][0] += 1
                emitfeature[predict_labels[i]]['UNK'][0] += 1

    return tranfeature, emitfeature


"""
TRAIN THE MODEL

1. structured perceptron
"""
def strperceptron(x, y, word_dict, label_dict, tranfeature, emitfeature, times=10):
    # numbers of training, to optimize lamda
    k = 0

    # overall training interations
    for i in range(times):
        for j in range(len(x)):
            training_x = x[j].copy()
            training_y = y[j].copy()
            sentence_prediction = viterbi(x[j], word_dict, label_dict, tranfeature, emitfeature)
            tranfeature, emitfeature = updatefeature(training_x, word_dict, training_y, sentence_prediction, tranfeature, emitfeature)
            k += 1

    # calculate average weights
    ## transition
    for u in list(tranfeature):
        for v in tranfeature[u]:
            tranfeature[u][v][0] /= (k + 1)
    ## emission
    for label in list(emitfeature):
        for word in emitfeature[label]:
            emitfeature[label][word][0] /= (k + 1)

    return tranfeature, emitfeature


"""
GENERATE N SHUFFLES AND FEATURES
"""
def nshufflefeature(x, y, word_dict, label_dict, tranfeature, emitfeature, itr, shuffles):
    trans_list = []
    emit_list = []

    for i in range(shuffles):
        xi, yi = shuffle_xy(x, y)
        tranfeaturei, emitfeaturei = strperceptron(xi, yi, word_dict, label_dict, tranfeature, emitfeature, times=itr)
        trans_list.append(tranfeaturei)
        emit_list.append(emitfeaturei)

    final_trans, final_emit = shuffle_average(trans_list, emit_list)
    return final_trans, final_emit


"""
COMPUTE AVERAGE TRANSITION AND EMISSION FEATURES
"""
def shuffle_average(trans_list, emit_list):
    final_trans = trans_list[0].copy()
    final_emit = emit_list[0].copy()

    # for transition features
    for u in final_trans:
        for v in final_trans[u]:
            summ = 0
            nonzero = 0
            for i in range(len(trans_list)):
                if trans_list[i][u][v][0] != 0:
                    summ += trans_list[i][u][v][0]
                    nonzero += 1
            if summ != 0:
                final_trans[u][v][0] = summ / nonzero

    # for emition features
    for state in final_emit:
        for word in final_emit[state]:
            summ = 0
            nonzero = 0
            for i in range(len(emit_list)):
                if emit_list[i][state][word][0] != 0:
                    summ += emit_list[i][state][word][0]
                    nonzero += 1
            if summ != 0:
                final_emit[state][word][0] = summ / nonzero

    return final_trans, final_emit


"""
PREDICTION
"""
def prediction(test_x, word_dict, label_dict, tranfeature, emitfeature, outputfile):
    f = open(outputfile, 'w')

    for testsentence in test_x:
        predict_sentence = viterbi(testsentence, word_dict, label_dict, tranfeature, emitfeature)
        for i in range(len(predict_sentence)):
            f.write(testsentence[i] + ' ' + predict_sentence[i] + '\n')
        f.write('\n')
    return f.close()


"""
EVALUATION
"""
def evaluation(gold, predict):
    import evalResultHelper as helper

    gold_file = open(gold)
    prediction_file = open(predict)
    observed = helper.get_observed(gold_file)
    predicted = helper.get_predicted(prediction_file)
    helper.compare_observed_to_predicted(observed, predicted, discardInstance=[])


"""
MAIN FUNCTION
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('-d', type=str, dest='data', help='Data set including traing, testing file.', required=True)
    parser.add_argument('-i', type=int, dest='i', help='Number of iterations.', default=10, required=False)
    parser.add_argument('-k', type=int, dest='k', help='Minimum number of appearances.', default=1, required=False)
    parser.add_argument('-s', type=int, dest='s', help='Number of shuffles.', default=5, required=False)

    # args = parser.parse_args(['-t', 'FR', '-i', '10', '-k', '1', '-s', '5'])
    args = parser.parse_args()

    trainingdata = '%s/train' % (args.data)
    testingdata = '%s/test.in' % (args.data)
    testingout = '%s/test.p5.out' % (args.data)
    goldfile = '%s/dev.out' % (args.data)

    x, y, word_dict, label_dict = readtrainingdata(trainingdata)
    tranfeature, emitfeature = constructfeature(x, word_dict, label_dict, k=args.k)

    shuffles = args.s
    final_trans, final_emit = nshufflefeature(x, y, word_dict, label_dict, tranfeature, emitfeature, args.i, shuffles)

    test_x = read_test(testingdata)
    prediction(test_x, word_dict, label_dict, final_trans, final_emit, testingout)

    # evaluation(goldfile, testingout)
