# 1) List (built in data type to store result of value)
#in python list we can store data of diffrent data types
marks = [90.2, 91.2, 93, 95]
print(marks)
print(type(marks))
print(marks[0])
print(len(marks))
print("\n")

#strings are immutabke (can not be changed)
#lists are mutable (can be changed)

marks[0]=40
print(marks)
print("\n")

# 2) slice
print(marks[0:2])
print("\n")

# 3) List methods

marks.append(80) # append value in list at the last
marks.sort() #ascending order sort
marks.sort(reverse=True) #descending order sort
marks.reverse() #reverse whole list in origional array
marks.insert(1,30) #insert(index, element)
print(marks)
print("\n")

# 4) Tuples
# builtin data types that let  us create [immutable] sequence of variables

tup =(5,4,2,1)
print(tup[0])
print(type(tup))
print("\n")

#if one va;ue add , after it otherwise type will be int,float,string depending upon the data typr

tup1 =(5) #Wrong
tup2 =(2,) #true
print(tup1)
print(tup2[0])
print(type(tup1))
print(type(tup2))
print("\n")


#chack Palindrome

list = [1,"abc","abc",1]
list1 = list.copy()
list1.reverse()

if (list == list1):
    print("list is a palindrome")
else:
    print("list is not a palindrome")