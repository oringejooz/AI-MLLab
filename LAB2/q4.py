# a program to print the multiplication table of a number
num = int(input("Enter number for table generation : "))
for i in range(1,11):
    print(f"{num} x {i} = {num*i}")