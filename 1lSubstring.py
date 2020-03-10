s = []
n = int(input("Enter number of elements : "))

for i in range(0, n):
    num = int(input())

    s.append(num)

print(s)
def SubSet(s, n):

    l = []
    for i in range(2 ** n):
        subset = ""
        for j in range(n):
            if (i & (1 << j)) != 0:
                subset += str(s[j]) + "|"
        if subset not in l and len(subset) > 0:
            l.append(subset)


    for subset in l:


        s = subset.split('|')
        for string in s:
            print(string, end=" ")

SubSet(s, n)

