# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:35:26 2021

@author: User
"""

print("Life is full of suprices and miracles")

#string
str1='what is change?'
str2='change is self analysis process.'
str1
str2
print(str1[0:10])
str1[0:10]
print(str1[6])
print(str2[15:])
str2[10:]
str1+str2
#string function
len(str1)    
len(str2)
str1.lower()
str2.upper()
str1.count("a")
str1.find("is")
#-------------------------------------------------------------------------------------------

#list in python
#list are mutable and enclosed with []
l1=[1,'amol','zore',True,100.1]
print(l1[1:3])
print(l1)
print(l1*2)
type(l1)
#lets try mutation
l1[2]='Tukaram'
print(l1)
#---------------------------------------------------------------------------------------------

#tuple in python 
#tuples are immutable 
#tuples enclosed with ()
tup=(1,8.3,False,'hello',4-2j)
tup
tup[3:4]
tup[3]
tup[3:]
tup[-1]
print(tup)
type(tup)
#--------------------------------------------------------------------------------------------

#dictionory in python
mumbai_indians11={1:'Rohit',2:'De cock',3:'suryakumar',4:'Ishan',5:'krunal',6:'Hardik',7:'Pollard',8:'Bolt',9:'rahul',10:'bumrah',11:'piyush'}
mumbai_indians11
print("captain of a team is",mumbai_indians11[1])
print("wicket kipper of a team is",mumbai_indians11[4])
print(mumbai_indians11[7],"is a vice captain of MI")
print(mumbai_indians11[1],"and",mumbai_indians11[2],"is a best opener in ipl 2020")
print(mumbai_indians11)
mumbai_indians11.keys()
mumbai_indians11.values()

#........ADVANCED..........
#List
top_ipl_team=["MI","CSK","RR","RCB"]
top_ipl_team
len(top_ipl_team)

#add item to list
top_ipl_team.append("SRH")
top_ipl_team

#sort
top_ipl_team.sort()
top_ipl_team

#delete item
del top_ipl_team[3]
top_ipl_team

#----------------------------------------------------------------------------------------------

#tuple
student=('virat','pujra','rahane','rahul')
student
type(student)
Data=1,4.5,"bcci","icc"
type(Data)

#--------------------------------------------------------------------------------------------

#dictionory
employee={'A01':'UMESH','A02':'RAHUL','A03':'SAGAR','A04':'NIKHIL'}
employee
employee['A03']
print("leader of team is",employee['A01'])
del employee['A04']
employee

#lets test mutation
#adding new element
employee['A104']='Balaji'
employee

#------------------------------------------------------------------------------------------

#set
Rohit={36,12,83,21}
Rohit
type(Rohit)
virat_score=set()
virat_score

Group_A=set(['IND','PAK','NZ','WI'])
Group_B=set(['ENG','AUS','SL','SA'])
Group_A
Group_B

#Checking whether a data value exists in a set or not.
'IND' in Group_A
'IND' in Group_B

#adding value in set
Group_A.add('AFG')
Group_B.add('IRE')
Group_A
Group_B

#create Dataframe :
    
import pandas as pd
high_score={'Name':['Rohit','Rahul','Pujara','Virat'],
             'Opponent':['SA','ENG','AUS','BAN'],
             'Venue':['Kolkatta','Chennai','Delhi','Mumbai'],
             'Score':[212,199,212,254]}
high_score

df=pd.DataFrame(high_score)
df

df['Score']
df[['Venue','Score','Opponent']]
df[0:3]

