# program to find the sum of all even numbers between 1 and 100.

sum=0
for i in range(1,101):
    if(i%2==0):
        sum+=i

print(f"Sum of numbers 1 to 100 is {sum}")