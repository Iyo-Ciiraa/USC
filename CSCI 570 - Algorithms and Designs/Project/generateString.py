def generateStringFromFile(inputFile):
    input_string_1, input_string_2, indices_string_1, indices_string_2 = readInput(inputFile)
    string1 = generateString(input_string_1,indices_string_1)
    string2 = generateString(input_string_2,indices_string_2)
    return string1,string2


def readInput(inputFile):
    indices_string_1 = []
    indices_string_2 = []

    input = [line.rstrip() for line in open(inputFile, 'r')]

    input_string_1 = input[0]

    j = 0
    for i in range(1, len(input)):
        if not (input[i].isnumeric()):
            input_string_2 = input[i]
            j = i + 1
            break
        else:
            indices_string_1.append(int(input[i]))

    for i in range(j, len(input)):
        indices_string_2.append(int(input[i]))

    return input_string_1, input_string_2, indices_string_1, indices_string_2


def generateString(inputString,stringIndices):

    outputString = inputString

    for index in stringIndices:
        outputString = outputString[0:index+1] + outputString + outputString[index+1:]

    return outputString