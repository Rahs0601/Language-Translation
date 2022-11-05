def spiralTraverse(array):
    result = []
    startrow, endrow = 0, len(array) - 1
    startCol, endCol = 0, len(array[0]) - 1
    while startrow <= endrow and startCol <= endCol:
        for col in range(startCol, endCol + 1):
            result.append(array[startrow][col])
        for row in range(startrow + 1, endrow + 1):
            result.append(array[row][endCol])
        for col in reversed(range(startCol, endCol)):
            if startrow == endrow:
                break
            result.append(array[endrow][col])
        for row in reversed(range(startrow + 1, endrow)):
            if startCol == endCol:
                break
            result.append(array[row][startCol])
        startrow += 1
        endrow -= 1
        startCol += 1
        endCol -= 1
    return result

arr = [[12,16,20,24],[28,32,36,40],[44,48,52,56],[60,64,68,72]]
print(spiralTraverse(arr))