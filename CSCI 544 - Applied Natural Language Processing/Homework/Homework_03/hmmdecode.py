import sys
import json

model = json.loads(open('hmmmodel.txt', 'r', encoding = 'UTF-8').read())

transition_probabilities = model['Transition Probabilities']
emission_probabilities = model['Emission Probabilities']
tags = model['Tags']
most_probable_tag = model['Most Probable Tag Word']

dev_data = sys.argv[1]

in_file = open(dev_data, 'r', encoding = 'UTF-8').read()
lines = in_file.splitlines()
tagged = []

def line_tagging(markov_chain, words):
    current_state = len(words)
    current_tag = 'end'

    line_tag = ""

    i = len(words) - 1

    while i >= 0:
        line_tag = words[i] + "/" + markov_chain[current_state][current_tag]['backprop'] + " " + line_tag
        current_tag = markov_chain[current_state][current_tag]['backprop']
        current_state = current_state - 1
        i -= 1
    return line_tag

for i in lines:
    words = i.split()
    first_word = words[0]

    markov_chain = []
    markov_chain.append({})

    states = {}

    if first_word in emission_probabilities.keys():
        states = emission_probabilities[first_word]
    else:
        states = most_probable_tag

    for tag in states:
        if tag == 'start' or tag == 'end':
            continue
        elif first_word in emission_probabilities:
            emission_values = emission_probabilities[first_word][tag]
        else:
            emission_values = 1

        markov_chain[0][tag] = {}
        markov_chain[0][tag]['probability'] = emission_values * transition_probabilities[tag]['start']
        markov_chain[0][tag]['backprop'] = 'start'

    for j in range(1, (len(words) + 1)):
        if j == len(words):
            last_word = markov_chain[-1]
            states = last_word.keys()
            max_probability = {'probability': 0, 'backprop': ''}
            markov_chain.append({})

            for tag in states:
                if tag == 'end':
                    continue
                else:
                    prev_probability = markov_chain[-2][tag]['probability'] * transition_probabilities['end'][tag]

                if (prev_probability > max_probability['probability']):
                    max_probability['probability'] = prev_probability
                    max_probability['backprop'] = tag
            
            markov_chain[-1]['end'] = {}
            markov_chain[-1]['end']['probability'] = max_probability['probability']
            markov_chain[-1]['end']['backprop'] = max_probability['backprop']
        else:
            current_word = words[j]
            markov_chain.append({})

            if current_word in emission_probabilities:
                states = emission_probabilities[current_word]
            else:
                states = most_probable_tag

            for tag in states:
                if tag == 'start' or tag == 'end':
                    continue
                elif current_word in emission_probabilities:
                    emission_values = emission_probabilities[current_word][tag]
                else:
                    emission_values = 1

                max_probability = {'probability': 0, 'backprop': ''}

                for prev_tag in markov_chain[j-1]:
                    if prev_tag == 'start' or prev_tag == 'end':
                        continue
                    else:
                        prev_probability = markov_chain[j-1][prev_tag]['probability'] * emission_values * transition_probabilities[tag][prev_tag]

                    if prev_probability > max_probability['probability']:
                        max_probability['probability'] = prev_probability
                        max_probability['backprop'] = prev_tag
                
                markov_chain[j][tag] = {}
                markov_chain[j][tag]['probability'] = max_probability['probability']
                markov_chain[j][tag]['backprop'] = max_probability['backprop']

    tagged.append(line_tagging(markov_chain, words))

out_file = open('hmmoutput.txt', 'w', encoding = 'UTF-8')
for i in tagged:
    out_file.write(i + '\n')

out_file.close