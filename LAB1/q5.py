#WAP to print the numbers from 1 to 10 using both for and while loops
start=int(input("Start Range : "))
end=int(input("End range : "))
for i in range(start,end+1):
    print(i)
print(end="\n")
i=start
while(i<=end):
    print(i)
    i+=1
