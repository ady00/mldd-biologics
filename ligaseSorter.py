import os
import shutil

Fdirectory = os.fsencode("FASTA dataset") #Set this to whatever directory has the files
Rdirectory = os.fsencode("Sorted Sequences") #Set this to the directory where you want the correct files to be stored
goodFiles = []
count = 0

for file in os.listdir(Fdirectory):
    absolutePath = os.path.join(Fdirectory, file)
    with open(absolutePath) as theFile:
        text = theFile.readline().rstrip()
        if "ligase" in text:
            goodFiles.append(file.decode())
            shutil.copy(absolutePath, Rdirectory)
            count +=1
print(goodFiles)
print(count)
#with open("C:\ASDRP\ML-DD\PDB Sort\sequenceNames.txt", 'w') as f: #This part is if you want the names of the sequences as a text file if you are planning other stuff. If you do, set the directory to where you want the text file to be, and uncomment these lines
#    for seq in goodFiles:
#        f.write(seq)
        
        
