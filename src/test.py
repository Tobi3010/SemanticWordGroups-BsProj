dic = {"w1":"c1", "w2":"c2", "w3":"c1"}

lst1 = [dic[i] for i in dic]
lst2 = dic.values()
lst3 = [i for i in dic.values()]
print(lst1)
print("\n")
print(lst2)
print("\n")
print(lst3)
