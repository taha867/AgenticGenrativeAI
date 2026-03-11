# 1) print("Hello world")

name = "taha"
age=18
weight = 54.90
gender = True  # true or false for bollean but T and F should be capital
 
print(type(name)) # string
print(type(age))  # int
print(type(weight)) # float
print(type(gender)) # boolean
print("\n")

# 2) python is case sensitive language

num1 = 2
num2 = 5
sum = num1 + num2
print("sum of num1 and num2 is : ", sum)
print("\n")

# 3) ctrl + / used to comment multiple lines

#4) Type Conversion
#(automatically convert into other type)
num1 = 5
num2 = 6.5
sum=num1+num2
print("type converted into float automatically : ",sum)
print("\n")

# 5) Type casting
# (MAually convert type of a variable)
# Lets say u want to add a string and integer python will give u error but u can do it by type casting by manually doing it

a,b=1,"2"
sum= a + int(b)
print("Type cast string into int : ",sum)
print("\n")

# 6) Take input in python

name = input("Enter your name : ")
print(type(name))
print("Welcome", name)
print("\n")
#input always takes string as input, if u want an integer u can type cast it and get the value

nameing = int(input("Enter your name(Now it will be in intiger using typecasting) : "))
print(type(name))
print("Welcome", nameing)
print("\n")