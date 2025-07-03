import saspy
sas = saspy.SASsession()
#print(sas.submit("proc options option=userid; run;")['LOG'])
print(sas.workpath)
file_list = sas.dirlist('/path/to/directory')
for file in file_list:
    print(file)

sas.endsas()