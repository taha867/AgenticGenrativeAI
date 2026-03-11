#1)  Dictionary
# Dictionary are used to store data values in key : value pair 
# keys can only be those data types that can't be cchanges
# Dictionary are unordered
# Dictionary is immutable and only ha unique key

info = {
    "name" : "Muhammad Taha",
    "age" : 22,
    "isAdult": True,
    "Marks": 92.2,
    "subjects": ["C++", "python", "java"],
    "topics": ("dictionary","set")
}


print(type(info))
print(info)
info["name"] = "Ahmed"
print(info["name"])
print(info["topics"])
print("\n")


# Nested Dictionary 

student = {
    "name" : "ali raza",
    "marks": {
        "phy": 80,
        "che":90,
        "Comp":100
    }
}

print(student["marks"]["che"])
print("Length :" ,len(student))
print("\n")
print(student.keys())  # get top level keys
print(list(student.keys()))  # get top level keys, typecast it into list 
print("\n")
print(student.values())  # get all values
print(list(student.values()))  # get all values,  typecast it into list 
print("\n")

print(student.items())  # get all key value pairs
print(list(student.items()))  # get all key value pairs,  typecast it into list 
print("\n")

# we can also get value using get() method in dictionary diffrence is
# if error occurs all other lines after it doesnot execute that why we use .get() method 
print(student.get("marks1")) # this will give "none" response on invalid value
#print(student["marks1"]) # this will give error on invalid value 
print("\n")

student.update({"city": "lahore"}) # to update the dictionary 
print(student)
print("\n")

# 2)  Sets
# collection of unordered items
# each element in set is unique and immutable 
# set will ignore the dublicates value if u write it , no error

#set is mutable but its elements are immutable

collection = {1,2,3,4,4,5}
# collection = set () # empty set 

print(type(collection))
print(len(collection))
print(collection)
print("\n")

collection.add(7)
collection.add(1)
print(collection)
collection.remove(1)
print(collection)
print("\n")

collection.clear()
print(collection)
print("\n")