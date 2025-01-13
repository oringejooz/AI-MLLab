#Create a  list, access elements modify elements and perform  list operations

fruits =["Apple","Orange","Grapes","Kiwi","Watermelon","Cherry"]
print(f"Accessing elements of {fruits} using indexing :")
print(f"First fruit : {fruits[0]}")
print(f"Third fruit : {fruits[2]}")#n-1
print(f"Last fruit : {fruits[-1]}")

fruits[1]="Mango"
print(f"Modified list {fruits}")

fruits.remove("Grapes")
print(f"Modified list {fruits}")

print(f"LEngth of Fruits List = {len(fruits)}")

