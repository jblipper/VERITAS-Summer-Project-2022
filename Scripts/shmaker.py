import numpy as np
with open('/Users/justinlipper/bashscript/bashscript.sh') as f:
    t=f.read()
f.close()
with open('/Users/justinlipper/bashscript/parameters.sh') as g:
    p=g.readlines()
g.close()
for i in p:
    i2=''
    for char in i:
        if char != '"' and char != "'" and char != '[' and char != ']' and char != ' ' and char != '(' and char != ')':
            i2+=char
    ii=i2.split(',')
    #print(ii)
    #print(i)
    date=ii[0]
    data=ii[9]
    source=ii[1]
    tn=ii[2]
    dp=f'/raid/reedbuck/lundym/frb/d{date}/{date}-{source}-T{tn}.csv'
    op='/raid/biggams/jlipper/Significance_Cutoff/data'
    er=data
    ss=16
    nc=16
    tr=250
    bt=ii[4]
    et=ii[5]
    sr=1/1200
    wo=ii[7]
    wd=ii[3]
    nb=ii[8]


    bashfile=f'{t} -dp {dp} -op {op} -er {er} -tn {tn} -ss {ss} -nc {nc} -tr {tr} -bt {bt} -et {et} -sr {sr} -wo {wo} -wd {wd} -nb {nb}'
    fl=open(f'/Users/justinlipper/bashscript/files/bashfile_Run{data}-T{tn}.sh','w')
    fl.write(bashfile)
    fl.close()

#print(t)
#print(p)