list1 = dict()
NoOfValues = int(input("enter the number of keys and values:"))
for i in range(0,NoOfValues):
    data = input('Enter key & value  ')
    tem = data.split(' ')
    list1[tem[0]] = int(tem[1])

# Displaying the dictionary
for key, value in list1.items():
    print('key {}, value {}'.format(key, value))

list2 = dict()
n = int(input("enter the number of keys and values:"))
for i in range(0,n):
    Output = input('Enter key & value  ')
    te = Output.split(' ')
    list2[te[0]] = int(te[1])

# Displaying the dictionary
for key, value in list2.items():
    print('key {}, value {}'.format(key, value))

list1.update(list2)
print(list1)
A= sorted(list1.values())
print(A)

