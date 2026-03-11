#1) escape sequence character
# \n -> next line
# \t -> space

#2) CONCENTATION
str1 = "ali"
str2 =  " ahmed"
final= str1 + str2

print(final)
print("\n")

#3) Lenth of string 
name = "my name is taha"
length = len(name)
print(length)
print("\n")

# 4) indexing
# to acess specific chracter at specific index 
str = "Muhammad Taha"
print(str[0])
print("\n")

# 5) Slicing
# accessing parts of a string
# str[ starting index : ending index ]
# starting index included but endin inex is not included 

index = "Muhammad ali johar"
ans = index[0:8]
print(ans)
print("\n")

#we can also do silicing negatively in python starting from -1
ans1 = index[-5:]
print(ans1)
print("\n")

#6) string function

string = "my name is taha and i work in kwanso"
print(string.endswith("so")) # check what strings end with
print(string.capitalize()) # make new strinhg and capitalize first word of string
print(string.replace("a","o")) # make new strinhg and replace one word with other
print(string.find("taha")) # returns starting index of its first occurence
print(string.count("taha")) # returns occurences of prticular thing in strring
print("\n")


#7) conditional statement

age=2

if(age>=20):
    print("person is an adult")
elif(age>=13 and age<=19):
    print("person is a teenager")
else:
    print("person is a baby")


print("\n")
