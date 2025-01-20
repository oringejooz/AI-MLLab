person={
    "name":"Swarnima",
    "age":21,
    "city":"Dehradun"
}

print(f"Name : {person["name"]}")
print(f"Age : {person["age"]}")
print(f"City : {person["city"]}")

person["email"]="bishtswarnima@gmail.com"

print(f"Updated dictionary = {person}")

if "city" in person:
    print(f"The city {person["city"]} exists in dictionary")

keys = person.keys()
values = person.values()
print("Keys:", list(keys))
print("Values:", list(values))
