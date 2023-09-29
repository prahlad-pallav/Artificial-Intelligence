# Built-in Functions -> python have many built-in functions such as print
print("Hello World")

# comments -> this line of code is not going to execute
# comments are of two types -> one and multiple lines comment

'''
This is a 
multi lines comment
with triple quotes
'''

# variable -> basically containers for storing data.

# data types -> number, string, list, tuple, dictionary

num1 = 5
num2 = 10
result1 = num1 + num2
print(result1)

num3 = 1.4
num4 = 0.65
result2 = num3 / num4
print(result2)

num5, num6 = 3.4, 2
result3 = num5 - num6
print(result3)

# boolean -> True or False
# these boolean are very useful in conditional statements -> if-else statement

print(3 > 1)
print(3 < 1)

result4 = 1 > 4
print(type(result4))

# string -> one-dimensional array
# there is no character in python, characters are just string with a single letter
# we're able to acquire the given characters with the help of indices.

name = 'prahlad pallav'
print(name[0])
print(name[-1])

print(name[:3])
print(name[:])

# step size in string

print(name[0::2])
print(name[::-1])

# string concatenation

firstName = "prahlad"
lastName = "pallav"

result5 = firstName + lastName
print(result5)

# format() -> in order to concatenate string and integer

text = "This is just a text with some number {}"
num = 23

result6 = text.format(num)
print(result6)

# string reversal

string = "Reverse this string"
print(string[::-1])

# type casting -> sometimes we want to specify the type of variables
# As python is an object-oriented programming languages, so it defines the data types with classes

a = int(23)
print(type(a))

b = int('23')
print(type(b))

c = int(123.13241)
print(c)

d = float(12)
print(d)
print(type(d))

e = float("231")
print(e)

f = str(2121)
print(f)
print(type(f))

# arithmetic operators -> +, -, *, /
a = 10
b = 3

# integer division
print(a//b)

# exponential
print(a**b)

# comparison
print(3 <= 10)
print(3 == "3")
print(3 != "3")

# assignment

assign = 3
print(assign)

# we can combine assignment operator with arithmetic operator

assign += 1
print(assign)


# conditional statement -> if-else statement

age = 21

if age > 18:
    print("Yes, you can vote")
else:
    print("No, you're not eligible to vote")


# input() -> input from the user
# num7 = input("Please enter a number: ")
# print(num7)

# Multiple conditional statements -> if-elif statement

# num8 = int(num7)
#
# if num8 == 0:
#     print("The given number is zero")
# elif num8 % 2 == 0:
#     print("The given number is even")
# else:
#     print("The given number is odd")


# logical operators -> or + and
# we can use or, and operation in conditional statements

# num9 = num8
#
# if num9 == 0 or num9 % 2 == 0:
#     print("The given number is even")
# else:
#     print("The given number is odd")



num10 = 23

if(num10 > 18 and num10 < 60):
    print("Yes, you can work")
else:
    print("No, you can't work")

# Not logical operator -> reverse the actual logical state of the given variable

c = True
print(not c)


# for loop

# from 0 to 4
for number in range(5):
    print(number)

# from 2 to 9
for number in range(2, 10):
    print(number)

# from 0 to 10 with increment of 2
for number in range(0, 11, 2):
    print(number)

name1 = "prahlad pallav"

for character in name1:
    print(character)


name2 = "pallav prahlad"
print(len(name2))

index = 0

while index < len(name2):
    print(name2[index])
    index += 1

# nested loops -> loop inside other loops

count = 1

for outer_index in range(5):
    for index_index in range(3):
        print('outer_index ' + str(outer_index) + ' inner_index ' + str(index_index))
        print(count)
        count += 1


# enumerate -> return indexes and values together

animals = ['cat', 'dog', 'horse']

for index, value in enumerate(animals):
    print(str(index) + " " + str(value))


# break statement

for index in range(10):
    if(index == 5):
        break
    else:
        print(index)

# continue statement

for index in range(10):
    if(index == 5):
        continue
    else:
        print(index)


# infinite loop

counter = 0

while True:

    if counter == 100:
        break
    else:
        print(counter)
        counter += 1


# fibonacci number -> fib(n) = fib(n-1) + fib(n-2)

a, b = 0, 1

while a < 100:
    print(a)
    a, b = b, a+b




























