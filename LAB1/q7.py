#Manipulate strings using built in functions
text="Hello, Welcome to Python Programming "
length_of_string=len(text)
print(f"Length of string : {length_of_string}")

uppercase_string=text.upper()
print(f"Uppercase version : {uppercase_string}")

lowercase_string=text.lower()
print(f"Lowercase version : {lowercase_string}")

substring="Python"
print(f"Is {substring} present in the string? {text}")

words_list =text.split()
print(f"List of Words {words_list}")