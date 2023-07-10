import string
import sys
import collections

stopwords = [
    "i", 
    "me", 
    "my", 
    "myself", 
    "we", 
    "our", 
    "ours", 
    "ourselves", 
    "you", 
    "your", 
    "yours", 
    "yourself",
    "he", 
    "him", 
    "his", 
    "himself", 
    "she", 
    "her", 
    "hers", 
    "herself", 
    "it", 
    "its", 
    "itself", 
    "they", 
    "them", 
    "their", 
    "theirs", 
    "themselves", 
    "what", 
    "which", 
    "who", 
    "whom", 
    "this", 
    "that", 
    "these", 
    "those", 
    "am", 
    "is", 
    "are", 
    "was", 
    "were", 
    "be", 
    "been", 
    "being", 
    "have", 
    "has", 
    "had", 
    "having", 
    "do", 
    "does", 
    "did", 
    "doing", 
    "a", 
    "an", 
    "the", 
    "and", 
    "but", 
    "if", 
    "or", 
    "because", 
    "as", 
    "until", 
    "while", 
    "of", 
    "at", 
    "by", 
    "for", 
    "with", 
    "about", 
    "against", 
    "between", 
    "into", 
    "through", 
    "during", 
    "before", 
    "after", 
    "above", 
    "below", 
    "to", 
    "from", 
    "up", 
    "down", 
    "in", 
    "out", 
    "on", 
    "off", 
    "over", 
    "under", 
    "again", 
    "further", 
    "then", 
    "once", 
    "here",
    "there", 
    "when", 
    "where", 
    "so", 
    "than", 
    "too", 
    "very", 
    "can", 
    "will", 
    "just", 
    "should", 
    "now"
]

vocabulary = set()

def vanilla():

    file_input = open(sys.argv[1], 'r').read()
    lines = file_input.splitlines()

    global vocabulary

    weights_fake = dict()
    weights_positive = dict()
    class_fake = dict()
    class_positive = dict()

    features = collections.OrderedDict()

    for i in lines:
        i = i.replace("'", '')
        i = i.replace("-", '')

        for j in string.punctuation:
            i = i.replace(j, ' ')

        words = i.strip(' ').split()

        if words[1] == 'True':
            class_fake[words[0]] = 1
        else:
            class_fake[words[0]] = -1

        if words[2] == 'Pos':
            class_positive[words[0]] = 1
        else:
            class_positive[words[0]] = -1

        count = dict()

        for k in range(3, len(words)):
            if words[k].lower() not in stopwords:
                vocabulary.add(words[k].lower())

                try:
                    count[words[k].lower()] += 1
                except KeyError:
                    count[words[k].lower()] = 1

        features[words[0]] = count

    for word in vocabulary:
        weights_fake[word] = 0
        weights_positive[word] = 0

    return weights_fake, weights_positive, class_fake, class_positive, features


def averaged():
    file_input = open(sys.argv[1], 'r').read()
    lines = file_input.splitlines()

    global vocabulary

    weights_fake = dict()
    weights_positive = dict()
    weights_fake_c = dict()
    weights_positive_c = dict()
    class_fake = dict()
    class_positive = dict()

    features = collections.OrderedDict()

    for i in lines:
        i = i.replace("'", '')
        i = i.replace("-", '')

        for j in string.punctuation:
            i = i.replace(j, ' ')

        words = i.strip('\n').split(' ')

        if words[1] == 'True':
            class_fake[words[0]] = 1
        else:
            class_fake[words[0]] = -1

        if words[2] == 'Pos':
            class_positive[words[0]] = 1
        else:
            class_positive[words[0]] = -1

        count = dict()

        for k in range(3, len(words)):
            if words[k] != '' and words[k].lower() not in stopwords:
                vocabulary.add(words[k].lower())

                try:
                    count[words[k].lower()] += 1
                except KeyError:
                    count[words[k].lower()] = 1

        features[words[0]] = count

    for word in vocabulary:
        weights_fake[word] = 0
        weights_positive[word] = 0
        weights_fake_c[word] = 0
        weights_positive_c[word] = 0

    return weights_fake, weights_positive, class_fake, class_positive, features, weights_fake_c, weights_positive_c


def vanilla_model():
    weights_fake, weights_positive, class_fake, class_positive, features = vanilla()

    bias_fake = 0
    bias_positive = 0

    for i in range(0,23):

        for j in features:
            feature = features[j]

            activation_fake = 0
            activation_positive = 0

            for word, count in feature.items():
                activation_fake += (weights_fake[word] * count)
                activation_positive += (weights_positive[word] * count)

            activation_fake += bias_fake
            activation_positive += bias_positive

            if (class_fake[j] * activation_fake) <= 0:
                for word, count in feature.items():
                    weights_fake[word] += (count * class_fake[j])

                bias_fake += class_fake[j]

            if (class_positive[j] * activation_positive) <= 0:
                for word, count in feature.items():
                    weights_positive[word] += (count * class_positive[j])

                bias_positive += class_positive[j]

    return bias_fake, bias_positive, weights_fake, weights_positive


def average_model():
    weights_fake, weights_positive, class_fake, class_positive, features, weights_fake_c, weights_positive_c = averaged()

    bias_fake = 0
    bias_positive = 0
    beta_fake = 0
    beta_positive = 0

    x = 1

    for i in range(0,23):

        for j in features:
            feature = features[j]

            activation_fake = 0
            activation_positive = 0

            for word, count in feature.items():
                activation_fake += (weights_fake[word] * count)
                activation_positive+= (weights_positive[word] * count)

            activation_fake += bias_fake
            activation_positive += bias_positive

            if (class_fake[j] * activation_fake) <= 0:
                for word, count in feature.items():
                    weights_fake[word] += (count * class_fake[j])
                    weights_fake_c[word] += (x * count * class_fake[j])

                bias_fake += class_fake[j]
                beta_fake += (class_fake[j] * x)

            if (class_positive[j] * activation_positive) <= 0:
                for word, count in feature.items():
                    weights_positive[word] += (count * class_positive[j])
                    weights_positive_c[word] += (x * count * class_positive[j])

                bias_positive += class_positive[j]
                beta_positive += (class_positive[j] * x)

            x += 1

    for val in weights_fake:
        weights_fake[val] -= float(weights_fake_c[val]) / x

    for val in weights_positive:
        weights_positive[val] -= float(weights_positive_c[val]) / x

    bias_fake -= float(beta_fake) / x
    bias_positive -= float(beta_positive) / x

    return bias_fake, bias_positive, weights_fake, weights_positive


def main():
    
    bias_fake, bias_positive, weights_fake, weights_positive = vanilla_model()

    output_file = open('vanillamodel.txt', 'w+')
    output_file.write("%s\n" % bias_fake)
    output_file.write("%s\n" % bias_positive)

    for word in vocabulary:
        output_file.write(word + " | weight_fake | %f \n" % (weights_fake[word]))
        output_file.write(word + " | weight_positive | %f \n" % (weights_positive[word]))

    output_file.close()

    bias_fake1, bias_positive1, weights_fake1, weights_positive1 = average_model()
    
    output_file = open('averagedmodel.txt', 'w+')
    output_file.write("%s\n" % bias_fake1)
    output_file.write("%s\n" % bias_positive1)

    for word in vocabulary:
        output_file.write(word + " | weight_fake | %f \n" % (weights_fake1[word]))
        output_file.write(word + " | weight_positive | %f \n" % (weights_positive1[word]))

    output_file.close()

if __name__ == '__main__':
    main()