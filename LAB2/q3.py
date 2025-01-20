# Lists and Tuples
list_1 = [2,4,6,8,10]
tuple_1=(2,4,6,8,10)
list_1[2]=10
print(f"Modified list : {list_1}")

try:
    tuple_1[1]=0
except TypeError as e:
    print(e)

print(f"Original tuple is {tuple_1}")
