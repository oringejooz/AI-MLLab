#WAP to check number is prime

num=int(input("Enter a number : "))
if(num>1):
    for i in range(2,(num//2)+1):
        if(num%i==0):
            print("Not Prime")
            break
        else:
            print("Prime")
else:
    print("Not Prime")

