
chars = {'A', 'B', 'C', 'D', 'E'}
dic = dict((i,j) for i,j in enumerate(chars))
print(dic)



dic.setdefault(1, 'QQQQQ')
print(dic)


dic[1] = 'Q'
print(dic)