
import time
import psutil
import sys


sys.setrecursionlimit(10000000)
DELTA = 30

def optimalStringSequence(str1,str2):
    string1, string2 = str1,str2
    ALPHA, OPT = initALPHA(), initOPT(DELTA, len(string1) + 1, len(string2) + 1)
    optimalValue = sequenceAlignmentValue(string1,string2,len(string1),len(string2),ALPHA,OPT)
    outputString1, outputString2 = optimalSequence(string1,string2,ALPHA,OPT)
    return optimalValue,outputString1,outputString2



def sequenceAlignmentValue(string1,string2,j,i,ALPHA,OPT):
    if(i==0 or j==0):
        return OPT[j][i]

    else:
        if (OPT[j - 1][i - 1] == 0):
            OPT[j - 1][i - 1] = sequenceAlignmentValue(string1,string2,j - 1, i - 1,ALPHA,OPT)

        if (OPT[j - 1][i] == 0):
            OPT[j - 1][i] = sequenceAlignmentValue(string1,string2,j - 1, i,ALPHA,OPT)

        if (OPT[j][i - 1] == 0):
            OPT[j][i - 1] = sequenceAlignmentValue(string1,string2,j, i - 1,ALPHA,OPT)

        OPT[j][i] = min((OPT[j - 1][i - 1] + ALPHA[charToAlphaIndex(string1[j-1])][charToAlphaIndex(string2[i-1])]),(OPT[j - 1][i] + DELTA),(OPT[j][i - 1] + DELTA))
        return OPT[j][i]

def optimalSequence(string1,string2,ALPHA,OPT):
    j = len(string1)
    i = len(string2)

    outputString1 = ""
    outputString2 = ""

    while(i!=0 and j!=0):

        if(j==0):
            outputString2 = "_"+outputString2
            i-=1
            continue
        elif(i==0):
            outputString1 = "_"+outputString1
            j-=1
            continue



        a = (OPT[j - 1][i - 1] + ALPHA[charToAlphaIndex(string1[j-1])][charToAlphaIndex(string2[i-1])])
        b = (OPT[j - 1][i] + DELTA)
        c = (OPT[j][i - 1] + DELTA)

        if(a<=b and a<=c):
            outputString1 = string1[j-1] + outputString1
            outputString2 = string2[i - 1] + outputString2
            j-=1
            i-=1

        elif(b<=a and b<=c):
            outputString1 = string1[j-1] + outputString1
            outputString2 = "_" + outputString2
            j-=1

        elif(c<=a and c<=b):
            outputString2 = string2[i - 1] + outputString2
            outputString1 = "_" + outputString1
            i -= 1

    return outputString1,outputString2


def basicSequenceAlignment(inputFile):
    string1, string2 = generateStringFromFile(inputFile)
    return optimalStringSequence(string1,string2)

def time_wrapper():
    start_time = time.time()
    optimalValue,outputString1,outputString2 = basicSequenceAlignment('input.txt')
    end_time = time.time()
    time_taken = (end_time - start_time)*1000
    return time_taken,optimalValue,outputString1,outputString2

def process_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_consumed = int(memory_info.rss/1024)
    return memory_consumed

def basicOutput():
    time_taken, optimalValue, outputString1, outputString2 = time_wrapper()
    memory = process_memory()
    outputFile = open('output.txt', 'w+')
    outputFile.write(str(optimalValue) + "\n" + outputString1 + "\n" + outputString2 + "\n" + str(time_taken) + "\n" + str(memory))
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
    
basicOutput()