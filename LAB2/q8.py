# a simple calculator using conditional statements and loops
while True:

    
    print("Choose an operation : ")
    print("1. Additon ")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Division")
    print("5. Modulus")
    print("6. Exit")

    choice=input("Enter your choice : ")

    num1=int(input("Enter first number : "))
    num2=int(input("Enter first number : "))
    if choice == '1':
        print(f"The result of {num1} + {num2} is {num1 + num2}")
    elif choice == '2':
        print(f"The result of {num1} - {num2} is {num1 - num2}")
    elif choice == '3':
        print(f"The result of {num1} * {num2} is {num1 * num2}")
    elif choice == '4':
        if num2 != 0:
            print(f"The result of {num1} / {num2} is {num1 / num2}") 
        else:
            print("Error: Division by zero is undefined.")
    elif choice == '5':
        if num2 != 0:
            print(f"The result of {num1} % {num2} is {num1 % num2}")
        else:
            print("Error: Modulus by zero is undefined.")
    elif choice == '6':
        print("Exiting the calculator. Goodbye!")
        break
    else:
        print("Invalid choice. Please select a valid operation.")
    
