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
    