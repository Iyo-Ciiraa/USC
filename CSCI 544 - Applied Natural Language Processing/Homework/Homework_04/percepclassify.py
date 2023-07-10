import sys
import string
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
    "why", 
    "how", 
    "all", 
    "any", 
    "both", 
    "each", 
    "few", 
    "more", 
    "most", 
    "other", 
    "some", 
    "such", 
    "only", 
    "own", 
    "same", 
    "so", 
    "than", 
    "too", 
    "very", 
    "can", 
    "will", 
    "just", 
    "should", 
    "now", 
    'the', 
    'for', 
    'had', 
    'and', 
    'to', 
    'a', 
    'was', 
    'in', 
    'of', 
    'you', 
    'is', 
    'it', 
    'at', 
    'with', 
    'they', 
    'on', 
    'our', 
    'be', 
    'as', 
    'there', 
    'an', 
    'or', 
    'this', 
    'my', 
    'that'
    ]

def read_model():
    input_file = open(sys.argv[1], 'r').read()
    lines = input_file.splitlines()

    bias_fake = float(lines[0])
    bias_positive = float(lines[1])

    weight_fake = dict()
    weight_positive = dict()

    for i in lines[3:]:
        i = i.split(" ")
        word = i[0]
        weight_type = i[2]
        weight = i[4]
        if weight_type == "weight_fake":
            weight_fake[word] = float(weight)
        if weight_type == "weight_positive":
            weight_positive[word] = float(weight)

    return bias_fake, bias_positive, weight_fake, weight_positive


def perceptron_model():
    bias_fake, bias_positive, weight_fake, weight_positive = read_model()
    input_file = open(sys.argv[2],'r').read()
    lines = input_file.splitlines()

    out = collections.OrderedDict()

    for i in lines:
        i = i.replace("'", '')
        i = i.replace("-", '')

        for j in string.punctuation:
            i = i.replace(j, ' ')

        words = i.strip("\n").split(" ")

        val = words[0]

        temp = list()
        for k in range(1, len(words)):
            if words[k] != '' and words[k].lower() not in stopwords:
                temp.append(words[k].lower())
                out[val] = temp

    output_file = open("percepoutput.txt", 'w+')

    for key, value in out.items():
        count = dict()
        for word in value:
            try:
                count[word] += 1
            except KeyError:
                count[word] = 1

        val_fake = 0
        val_positive = 0

        for word, count in count.items():
            if word in weight_fake:
                val_fake += (weight_fake[word] * count)

            if word in weight_positive:
                val_positive += (weight_positive[word] * count)

        val_fake += bias_fake
        val_positive += bias_positive

        if val_fake > 0:
            f = 'True'
        else:
            f = 'Fake'

        if val_positive > 0:
            s = 'Pos'
        else:
            s = 'Neg'

        output_file.write(key + " " + f + " " + s + "\n")

def main():
        perceptron_model()

if __name__ == '__main__':
    main()