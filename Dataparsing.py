import os


directory = os.fsencode("FASTA dataset")
goodfiles = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    sequence = open(filename)
    if("ligase" in sequence.read()):
        goodfiles+=filename
