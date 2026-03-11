# 1) functions


def calculateSum(a,b):
    sum =a + b
    return sum


a = int(input("Enter first value :"))
b = int(input("Enter second value :"))

sum = calculateSum(a,b)
print(sum)
print("\n")

# 2) Recurrsion
# WHEN A FUNCTION CALLS ITSELF

# EX#01) print num backward
def show(n):
    if (n == 0 ):
        return

    print(n)
    show(n - 1)

show(5)
print("\n")

# EX#02) Factorial

def factorial(n):
    if(n == 0 or n ==1):
        return 1
    else:
        return factorial(n-1)*n

print(factorial(5))
print("\n")