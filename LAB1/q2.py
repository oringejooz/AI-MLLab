#Create variables of diff datatypes and perform basic operations on them

integer=10
float=50
string="Hello"
boolean=True
#Performing arithmetic operations on numeric data types
add = integer+float
subt=integer-float
mult=integer*float
div=integer/float
modu=integer%float

print(f"additon:{integer}+{float}={add}")
print(f"subtraction:{integer}-{float}={subt}")
print(f"multiplication:{integer}*{float}={mult}")
print(f"division:{integer}/{float}={div}")
print(f"modulus:{integer}%{float}={modu}")

concatenated_string=string+"World!!"
print(concatenated_string)

boolean=(bool(input("Enter True or False : ")))
andop=boolean and False
orop=boolean or False
notop=not boolean
print(andop)
print(orop)
print(notop)

