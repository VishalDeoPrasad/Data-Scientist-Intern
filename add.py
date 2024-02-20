addition.py
def add(a,b):
    return a+b

API.py
import addition as add
a = int(input("enter a"))
b = int(input("enter b"))
add.add(a,b)
