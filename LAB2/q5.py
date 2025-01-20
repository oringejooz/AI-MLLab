#Func to find factorial of a number

def factorial(n):
    if(n<0):
        print("Negative num factorial undefined")
    elif (n==0):
        return 1
    else:
        res=1
        for i in range(1,n+1):
            res*=i
        return res


print(f"Factorial of 5 is {factorial(5)}")