# open, read and close files

# 1) open a file

# f = open("file_name", "mode")
#2) read file
f = open("demo.txt","r")
data = f.read()
print(data)
print(type(data))
f.close()
print("\n")

#  'r' -> open for reading default
#  'w' -> open for writing, truncuating the file first  ()it means 'overwrite', all the data is deleted then ju write
#  'x' -> create a new file and open it for writing 
#  'a' -> open for writing, appending to end of the file if it exists 
#  'b' -> binary mode 
#  't' -> text mode 
#  '+' -> open a disk file for updating reading and writing 

#3) read file line by line
f = open("demo.txt","r")
data = f.readline()
print(data)
print(type(data))
f.close()
print("\n")

