import sys
import json

model_file = "hmmmodel.txt"

train_data = sys.argv[1]

in_file = open(train_data, encoding = 'UTF-8')

words = [line.rstrip('\n').split() for line in in_file]

tags = {}
word_tag = {}
most_probable_tag = {}

tags['start'] = len(words)
tags['end'] = len(words)

for i in words:
    for t in i:
        mid = t.split('/')
        tag = mid[-1]
        word = mid[:-1]

        if tag not in most_probable_tag:
            most_probable_tag[tag] = []

        if word[0] not in most_probable_tag[tag]:
            most_probable_tag[tag].append(word[0])

        word = '/'.join(word)

        if tag not in tags:
            tags[tag] = 1
        else:
            tags[tag] +=1

        if tag not in word_tag:
            word_tag[tag] = {}
        
        if word not in word_tag[tag]:
            word_tag[tag][word] = 1
        else:
            word_tag[tag][word] += 1

most_probable_tag = sorted(most_probable_tag, key = lambda x: len(most_probable_tag[x]), reverse = True)

out_file = open('hmmmodel.txt', 'w')

emission_probabilities = {}

for tag in word_tag:
    for word in word_tag[tag]:
        if word not in emission_probabilities:
            emission_probabilities[word] = {}
        if tag not in emission_probabilities[word]:
            emission_probabilities[word][tag] = word_tag[tag][word]/tags[tag]

transition_probabilities = {}

for tag in tags:
    transition_probabilities[tag] = {}

for w in words:
    for i in range(len(w) + 1):
        if i == 0:
            if 'start' not in transition_probabilities[w[i].split('/')[-1]]:
                transition_probabilities[w[i].split('/')[-1]]['start'] = 1
            else:
                transition_probabilities[w[i].split('/')[-1]]['start'] += 1
        elif i == len(w):
            if w[i-1].split('/')[-1] not in transition_probabilities['end']:
                transition_probabilities['end'][w[i-1].split('/')[-1]] = 1
            else:
                transition_probabilities['end'][w[i-1].split('/')[-1]] += 1
        else:
            if w[i-1].split('/')[-1] not in transition_probabilities[w[i].split('/')[-1]]:
                transition_probabilities[w[i].split('/')[-1]][w[i-1].split('/')[-1]] = 1
            else:
                transition_probabilities[w[i].split('/')[-1]][w[i-1].split('/')[-1]] += 1

for i in transition_probabilities:
    for j in tags:
        if j == 'end':
            continue
        elif j not in transition_probabilities[i]:
            transition_probabilities[i][j] = 1/(tags[j] + (4*len(tags)) - 1)
        else:
            transition_probabilities[i][j] = (transition_probabilities[i][j] + 1)/(tags[j] + (4*len(tags)) - 1)

model = {'Tags': tags, 'Transition Probabilities': transition_probabilities, 'Emission Probabilities': emission_probabilities, 'Most Probable Tag Word': most_probable_tag[0:5]}

out_file.write(json.dumps(model, indent = 2))

out_file.close
in_file.close