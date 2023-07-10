import sys 
import time
import psutil 

def timeWrapper():

    startTime = time.time()
    outputString1, outputString2, optimalValue = efficientAlgorithm()
    endTime = time.time()
    timeTaken = (endTime - startTime) * 1000

    return outputString1, outputString2, optimalValue, timeTaken

def processMemory():

    process = psutil.Process()
    memoryInfo = process.memory_info()
    memoryConsumed = int(memoryInfo.rss/1024)
    return memoryConsumed

def basicOptimalSequence(string1, string2):

    lenStr1, lenStr2 = len(string1), len(string2)
    OPT = []

    for i in range(lenStr1 +1):
        OPT.append([0]*(lenStr2+1))
    
    for j in range(lenStr2+1):
        OPT[0][j] = DELTA*j
    
    for i in range(lenStr1+1):
        OPT[i][0] = DELTA*i

    for i in range(1,lenStr1+1):
        for j in range(1,lenStr2+1):
            OPT[i][j] = min(OPT[i-1][j-1] + ALPHA[charToAlphaIndex(string1[i-1])][charToAlphaIndex(string2[j-1])],
                            OPT[i-1][j] + DELTA,
                            OPT[i][j-1] + DELTA)
    
    outputString1 = ""
    outputString2 = ""

    i, j = lenStr1, lenStr2

    while i and j:
        if OPT[i][j] == OPT[i-1][j-1] + ALPHA[charToAlphaIndex(string1[i-1])][charToAlphaIndex(string2[j-1])]:
            outputString1 = string1[i-1] + outputString1
            outputString2 = string2[j-1] + outputString2
            i = i-1
            j = j-1
        elif OPT[i][j] == OPT[i-1][j] + DELTA:
            outputString1 = string1[i-1] + outputString1
            outputString2 = '-' + outputString2
            i = i-1
        elif OPT[i][j] == OPT[i][j-1] + DELTA:
            outputString1 = '-' + outputString1
            outputString2 = string2[j-1] + outputString2
            j = j-1
    
    while i:
        outputString1 = string1[i-1] + outputString1
        outputString2 = '-' + outputString2
        i = i-1

    while j:
        outputString1 = '-' + outputString1
        outputString2 = string2[j-1] + outputString2
        j = j-1

    return outputString1, outputString2, OPT[lenStr1][lenStr2]

def forwardCheck(string1, string2):

    lenStr1, lenStr2 = len(string1), len(string2)
    OPT = []
    
    for i in range(lenStr1+1):
        OPT.append([0]*(lenStr2+1))
    
    for j in range(lenStr2+1):
        OPT[0][j] = DELTA*j
    
    for i in range(1, lenStr1+1):
        OPT[i][0] = OPT[i-1][0] + DELTA
        for j in range(1, lenStr2+1):
            OPT[i][j] = min(OPT[i-1][j-1] + ALPHA[charToAlphaIndex(string1[i-1])][charToAlphaIndex(string2[j-1])],
                            OPT[i-1][j] + DELTA,
                            OPT[i][j-1] + DELTA)
        OPT[i-1] = []
    return OPT[lenStr1]    

def reverseCheck(string1, string2):

    lenStr1, lenStr2 = len(string1), len(string2)
    OPT = []
    
    for i in range(lenStr1+1):
        OPT.append([0]*(lenStr2+1))
    
    for j in range(lenStr2+1):
        OPT[0][j] = DELTA*j
    
    for i in range(1, lenStr1+1):
        OPT[i][0] = OPT[i-1][0] + DELTA
        for j in range(1, lenStr2+1):
            OPT[i][j] = min(OPT[i-1][j-1] + ALPHA[charToAlphaIndex(string1[lenStr1-i])][charToAlphaIndex(string2[lenStr2-j])],
                            OPT[i-1][j] + DELTA,
                            OPT[i][j-1] + DELTA)
        OPT[i-1] = []
    return OPT[lenStr1]

def efficientOptimalSequence(string1, string2):

    lenStr1, lenStr2 = len(string1), len(string2)
    
    if lenStr1<2 or lenStr2<2:
        return basicOptimalSequence(string1, string2)
    else:
        midStr1 = int(lenStr1/2)
        F, B = forwardCheck(string1[:midStr1], string2), reverseCheck(string1[midStr1:], string2)
        partition = [F[j] + B[lenStr2-j] for j in range(lenStr2+1)]
        midStr2 = partition.index(min(partition))

        F, B, partition = [], [], []

        callLeft = efficientOptimalSequence(string1[:midStr1], string2[:midStr2])
        callRight = efficientOptimalSequence(string1[midStr1:], string2[midStr2:])

        return [callLeft[r] + callRight[r] for r in range(3)]

def efficientAlgorithm():

    string1, string2 = generateStringFromFile("input.txt")
    return (efficientOptimalSequence(string1, string2))

def efficientOutput():

    memoryConsumed = processMemory()
    outputString1, outputString2, optimalValue, timeTaken = timeWrapper()

    outputFile = open('output.txt', 'w')
    outputFile.write(str(optimalValue) + "\n" + outputString1 + "\n" + outputString2 + "\n" + str(timeTaken) + "\n" + str(memoryConsumed))
    outputFile.close()

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

def initOPT(delta,rows,cols):
    OPT = [[0 for i in range(cols)] for j in range(rows)]

    for j in range(0, rows):
        for i in range(0, cols):
            if j == 0:
                OPT[j][i] = i * delta
            elif i == 0:
                OPT[j][i] = j * delta

    return OPT

def initALPHA():
    ALPHA = [[0, 110, 48, 94], [110, 0, 118, 48], [48, 118, 0, 110], [94, 48, 110, 0]]
    return ALPHA

def charToAlphaIndex(character):
    switch={
        'A':0,
        'C':1,
        'G':2,
        'T':3
    }
    return switch.get(character,"Invalid Input")

DELTA = 30
ALPHA = initALPHA()
efficientOutput()