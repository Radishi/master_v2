

num = list(range(92))
expect = [25,26,27,28,29,30,31,32,33,34,35,36,37,39,40,41,43,46,49,50,52,53,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,82,84,85,86,87,88,89,90,91]
l = 0
index = 1
for i in range(91):
    if num[index]==expect[l]:
        num.pop(index)
        l += 1
    else:
        index += 1
print(len(num))
add = {6:"a",7:"b",25:"c",26:"d",27:"e"}
for i,v in add.items():
    num.insert(i,v)

# num.insert(1,999)
for i,n in enumerate(num):
    print(" ",i,"<--",n,end=' |')
    # print(n,end=',')



