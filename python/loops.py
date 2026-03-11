# Loops

# 1) While loop
count = 0
while(count<5):
    print("Hello")
    count+=1
print("\n")
# search a number in tuple

# 2) break
nums = ( 20 , 40 , 1 , 68 , 79 , 100, 54, 3, 7 )

i = 0
x = int(input("Enter your number:"))

while(i<len(nums)):
    if(nums[i] == x):
        print("Number found at index : ", i)
        break # breaks the loop 
    else:
        print("finding...")
        
    i+=1

print("End of Loop")
print("\n")

# 3) continue
j=1
while(j <= 5):
    if(j==3):
        j+=1
        continue # skip all steps for this particular iteration
    else:
        print(j)
        j+=1
        
print("\n")       
# 4) For loop

lists = [1, 2, 3, 4, 5,8,5]

for val in lists:
    print(val)

print("\n")

# else in for loop

str = "Muhammad Taha"
k=0
for char in lists:
    if(char == 5):
        lists[k] = "@"
        k+=1
        continue
    else:
        k+=1
        print("Processing...........!")
else:
    print("whole loop ran")

print(lists)
print("\n")  

# 5) Range
# Range function returns a sequence of numbers starting from 0 by default, and increments by 1 by default
# and stops before a specific number
# range( start?, stop, step?)

for el in range(5,20,2):
    print(el)
