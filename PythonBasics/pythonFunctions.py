
# parameters -> passed inside function during formation, here name is a parameter.
def show_name(name):
    age = "21"
    print("The name of the person is : " + name)
    print("The age of the person is : " + age)

# arguments -> passed inside function during function call. here prahlad is an argument.
show_name("prahlad")

def show_info(name, age):
    print("The name of the person is : " + name)
    print("The age of the person is : " + age)

# arguments -> passed inside function during function call. here prahlad is an argument.
show_info("pallav", "23")

def show_pass():
    pass

# positional arguments -> the order of the argument does matter

def show_name1(name, age):
    print("The name of the person is : " + name)
    print("The age of the person is : " + age)

show_name1("monu", "23")


# *number -> we use *, when we don't know number of argument to be passed.
def show_number(*number):
    # tuple -> one dimensional array so we can use indexes
    # print(number[0])
    # print(number[1])
    for index in range(len(number)):
        print(number[index])

show_number(23, 34)

# default value for the parameter
def square(n = 2):
    print(n * n)

square()
square(3)


# keyword argument -> in this case, order doesnot matter
def show_information(name, age):
    print(name)
    print(age)

show_information(name = "Ramu", age="21")
show_information(age="45", name="nikhil")

# ** -> when we don't know arbitrary number of keywords arguments -> it is a dictionary with key and value pairs
def show_informations(**params):
    print(params["name"])
    print(params["lastName"])

show_informations(name="devil", lastName="singh")


# return -> it return the value from the given functions

def subtract(num1, num2):
    res = num1 - num2
    return res

result = subtract(3, 5)
print(result)

# return multiple values

def operation(x):

    if x % 2 == 0:
        return True, 1, "Even Number"

    return False, -1

oper = operation(10)
print(oper)

# yield -> it can produce a sequence of values/objects.

def producer():

    for num in range(0, 10, 1):
        if num % 2 == 0:
            yield num

for i in producer():
    print(i)


# local variable and global variable

num1 = 45
def local_function():
    num = 15
    print(num)
    print(num1)

local_function()
print(num1)

num2 = 78

def local_function1():
    global num2
    num2 = 55

local_function1()
print(num2)

# built-in functions

print("This is inside a built-in function")

age = "23"
age = int(age)
print(type(age))

print(pow(2,3))


# main function -> as there is no entry point in python, there are so-called implicit variable such as __name__
# the __name__ is a special built in variable which evaluates to the name of the current module
# however if a module is being run directly then this __name__ get the value __main__

def say_hello():
    print("Hello from say_hello function")

# say_hello()

if __name__ == "__main__":
    say_hello()

