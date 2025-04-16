import os

# print(os.listdir(path="C:\\Users\\Patrick Myers\\Downloads\\archive\\FICTION\\FICTION"))
files = os.listdir(path="C:\\Users\\Patrick Myers\\Downloads\\archive\\FICTION\\FICTION")  # returns of type list of str
text = ""
for file in files:
    print(file)
    text += open(file="C:\\Users\\Patrick Myers\\Downloads\\archive\\FICTION\\FICTION\\" + file , mode="r").read()

# print(text)
with open("out.txt", 'w') as output:
    output.write(text)