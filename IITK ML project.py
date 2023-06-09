#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("ML is simple")
import numpy
print("nunmpy=",numpy.__version__)


# In[2]:


import matplotlib
print("matplotlib]=", matplotlib.__version__)


# In[4]:


import pandas
print("pandas=",pandas.__version__)


# In[5]:


import sklearn
print("sklearn=",sklearn.__version__)


# In[6]:


import scipy
print("scipy=",scipy.__version__)


# In[7]:


import seaborn
print("scipy=",seaborn.__version__)


# In[10]:


print("All AI  pluggings are installed and working successfully")


# In[14]:


import csv
filename=open('C:/Users/Lohithakshan/Downloads/indians-diabetes.data.csv')
reader=csv.reader(filename,delimiter=' ')
lines=list(reader)
print('No of rows: ',len(lines))


# In[6]:


import numpy
filename=open("C:/Users/Lohithakshan/Downloads/indians-diabetes.data.csv")
data=numpy.loadtxt(filename,delimiter=",")
filename.close()
print("Numpy loadtxt Size= ",data.shape)


# In[8]:


import numpy
import urllib.request
web_path= urllib.request.urlopen("https://goo.gl/QnHW4g")
dataset= numpy.genfromtxt(web_path,delimiter=",")
print("Shape of Data from URL=",dataset.shape)

print("Format of data dim=",dataset.ndim)
print(dataset)

print("\n\n\n")

for line in dataset:
    print( line )



# In[ ]:


#20-12-22


# In[1]:


for ch in 'KANPUR' :
    print('Hello=',ch)


# In[2]:


words=['India','Japan','USA']
for w in words:
    print(w,len(w))


# In[5]:


s=input("Enter a word:")
s=s.upper()
count=0

for a in s:
    if a=='A' or a=='E' or a=='I' or a=='O' or a=='U':
        count=count+1
print("Total vowels=",count)        
        


# In[ ]:


#append- add one element at the end
#extend-add more values 


# In[7]:


arr=[]
print("Array Before: ",arr)
count=0

while(True):
    s1=input("Enter any friend name: ")
    arr.append(s1)
    choice=input("Wish to add more?(y/n): ")
    if(choice=='n'):
        break
print("Array After: ",arr)

for name in arr:
    if len(name)<=3:
        count=count+1
        
print("Count of smaller ones: ",count)      
        

        
        


# In[8]:


arr=['Mary','had','a','little','lamb']
print( enumerate(arr)  )
print(list( enumerate(arr) ) )
print(tuple( enumerate(arr) ) )


# In[9]:


for i in range(1,11):
    print(i)
    
arr=[i*i for i in range(1,11)]
print(arr)


# In[10]:


arr=[1,2,3,4,5,6,7,8,9,10]
for i in arr:
    if(i%3==0):
        print(i*10)
        


# In[1]:


arr=[input("Enter number "+ str(n)+ "=" ) for n in range(1,6)]
print(arr)


# In[6]:


for num in range(1,11):
    print(num)
else:
    print('This is else block of for loop')


# In[8]:


for num in range(1,11):
    print(num)
    if(num==10):
        break
else:
    print('This is else block of for loop')
    


# In[10]:


s1=input("Enter a sentence: ")
alpha=input("Enter alphabet to search: ")
status=False
for i in s1:
    if(i==alpha):
        status =True
        break
if status==True:
    print(alpha,"is found")
else:
    print(alpha,"not found")
        
    
    


# In[11]:


num=int(input("Enter a no. "))
status=True
for a in range (2,num):
    if num%a==0:
        status=False
        break;
if(status==False):
    print("Not a prime number")
else:
     print("Is a prime number")
        


# In[ ]:


#21-12-22


# In[ ]:


#map function(convert)


# In[8]:


arr="111 222 333 444 555"
s1=arr.split()
print(s1)


# In[10]:


s2='tttrrryyy'
s3=set(s2)
s3


# In[13]:


d={'A':'JAVA','B':'J2EE','C':'ANDROID','D':'PYTHON','E':'HADOOP','KEY':'VALUE'}
print("Dictionary= ",d)
print(dir())
del d
print(dir())
#dictionary doesn't accept positional value


# In[17]:


d={'A':'JAVA','B':'J2EE','C':'ANDROID','D':'PYTHON','E':'HADOOP','KEY':'VALUE'}
d['F']='JAPAN'
print(d)


# In[21]:


d={'A':11,'B':22,'C':33,'D':44,'E':55}
print("d=",d)
data=input("Enter data to delete:")
if data in d:
    del d[data]
    print(data,"deleted successfully")
else:
    print(data,"not found to delete")
    
print("d=",d)    

d['Z']=d.get('Z',0)+1
print(d)


# In[20]:


d['Z']=d.get('Z',0)+1
print(d)


# In[23]:


d={'A':11,'B':22,'C':33,'D':44,'E':55}
print("d=",d)
key=input("Enter the key to add/update:")
d[key]=d.get(key,0)+1
print("d=",d)


# In[24]:


def group1():
    a=int(input("Enter First Number: "))
    b=int(input("Enter Second Number: "))
    c=a+b
    print("Addition Result= ",c)
    return

group1()
print("==========")
group1()


# In[ ]:


def show(empName,phone="1234567890",city="Lucknow",company="XYZ"):
    print("\nempName= ",empName)
    print("phone= ", phone)
    print("city= ",city)
    print("Company= ",company)
    return

#rule1
show("Chintu")
#rule2
show("Chintu","99887766") #positional arguments
#rule3
show(empName="Pappu",phone="8936523") #keyword argument
show(phone="8936523",empName="Pappu")
show(empName="Pappu",city="Kanpur")


# In[25]:


#file handling for data
fob=open('sensor.txt','w')
totalchars=fob.write("This is the first line")
fob.close()

print("Result=",totalchars)
print('Data saved successfully')


# In[37]:


#reading data from file
fob=open('sensor.txt','r')
print(fob.read(6))  #this i
print("Latest starting location: ",fob.tell())
print(fob.read()) #s first line
print("First: ",fob.closed)
fob.close()
print("Second: ",fob.closed)
print("==========")

#add multiple lines of data in sensor1.txt file manually
fob=open('sensor.txt','r')
print("Line 1=", fob.readline())
print("Line 2=", fob.readline())
fob.close()
print("==========")

#.
fob=open('sensor.txt','r')
arr=fob.readlines()
print("Data= \n",arr)
fob.close()
print("==========")

#to print the last line of the file

fob=open('sensor.txt','r')
myList=fob.readlines()
print(myList[len(myList)-1])

fob.close()


# In[ ]:


#Only in python ,dual indexing is allowed


# In[41]:


fob=open('sensor.txt','r')
myList=fob.readlines()
print (myList[len(myList)-1])

fob.close()


# In[ ]:


#update the file data
fob=open('sensor.txt','r')
myList=fob.readlines()
fob.close()

print("Original: ",myList[2])
myList[2]="This is updated third  line\n"
print("Updated:",myList[2])


# In[ ]:


#23-12-22


# In[1]:


#Rescale data(custom range between  1 and 5)

import pandas as pd
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

filename='indians-diabetes.data.csv'
hnames=['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe=pd.read_csv(filename,names=hnames) #contains both data and meta data
array=dataframe.values
#separate array into input and output components
X=array[ : ,0:8]  #[rows,cols]
Y=array[ : ,8]
scaler=MinMaxScaler( feature_range=(1,5) ) #range

#first method
rescaledX=scaler.fit_transformation(X)

#second method
scaler=scaler.fit(X)
rescaledX=scaler.transform(X)

#summarize transformed data
set_printoptions(precision=2)
print(rescaledX[0:30,:])

print("\n\nMean of the first column",end="")
print(np.mean(rescaledX[:,0]))


# from sklearn.preprocessing import MinMaxScaler
# import pandas as pd
# filename='indians-diabetes.data.csv'
# names=['preg','plas','pres','skin','test','mass','pedi','age','class']
# dataframe=pd.read_csv(filename,names=names)
# array=dataframe.values
# X=array[:,0:8]
# Y=array[:,8]
# binarizer=Binarizer(threshold=5)
# binaryX=binarizer.fit_transform(X)
# print(binaryX[0:30,:])
# 

# In[3]:


#24-12-22

import matplotlib.pyplot as plt
plt.plot([1,2,3,4,5],[1,2,3,4,5],'go-',label='line 1',linewidth=2)
plt.plot([1,2,3,4,5],[1,4,9,16,25],'rs--',label='line 2',linewidth=4)
plt.axis([0,6,0,26])
plt.legend(loc="upper left")
plt.show()


# In[9]:


import matplotlib.pyplot as plt
x1=[1,2,3]
y1=[1,2,3]
x2=[1,2,3]
y2=[1,4,9]
plt.plot(x1,y1,'o-',x2,y2,'s--',linewidth=7)
plt.axis([0,4,0,10])
plt.show()


# In[13]:


import matplotlib.pyplot as plt
x=[2,3,4,5,6,7]
y=[4,9,16,25,36,49]
plt.plot(x,y,marker='o',markerfacecolor='red',markersize=15,linestyle='dashed',color='blue')
plt.title("Number with Squared Values")
plt.xlabel('-----Numbers------->',fontsize=14,color='red')
plt.ylabel('-----Square------->',fontsize=14,color='blue')
plt.axis([1,8,2,51])
plt.grid(True)
plt.annotate('Square of 5',xytext=(3,40),xy=(5,25), arrowprops=dict(facecolor='black',shrink=.1))
plt.show()


# In[20]:


import matplotlib.pyplot as plt
import numpy as np
t=np.arange(0.0,5.0,0.2) #start,end,interval
print(t)
#red stars, blue squares and green dots
plt.plot(t,t,'r*-',
         t,t+3,'bs-',
         t,t+6,'g-', t,t+6,'ro',
         markersize=7)
plt.show()


# In[21]:


import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(311)
plt.plot([1,2,3])

plt.subplot(312)
plt.plot([4,5,6])

plt.subplot(313)
plt.plot([7,8,9])

plt.figure(2)
plt.plot([11,12,13])

plt.figure(1)
plt.subplot(311)
plt.title('Easy as 1,2,3')

plt.figure(3)
import numpy as np
x=np.arange(1,6)
y=x**2
plt.plot(x,y,'ro-')

plt.show()


# In[ ]:


import matplotlib.pyplot as plt
x1=[1,2,3,4,5]
y1=[2,3,2,3,4]
x2=[2,3,4]
y2=[5,5,5]
x3=[1,2,3,4,5]
y3=[6,8,7,8,7]

plt.scatter(x1,y1)
plt.scatter(x1,y1)


# In[22]:


import matplotlib.pyplot as plt
myLabels=['s1','s2','s3']
sections=[60,90,50]
myColors=['c','g','r']
plt.pie(sections,labels=myLabels,colors=myColors,
        startangle=45,
       explode=(0,0.1,0),
       autopct='%1.2f%%')
plt.title('Pie Chart Example')
plt.show()


# In[3]:


#25-12-22

a=10
print(type(a))
a=12.5
print(type(a))
a="IITK"
print(type(a))


# In[10]:


import numpy as np
ddarr=np.array([[1,2,3],[4,5,6]])
print("ddarr.ndim=",ddarr.ndim)
print("ddarr.shape=",ddarr.shape)

print("ddarr.size=",ddarr.size)
print("len(ddarr)=",len(ddarr))
print("ddarr.dtype=", ddarr.dtype)

print(ddarr)
print("**********************")
print("ddarr[0,1] - ",ddarr[0,1])
print("ddarr[0] - ",ddarr[0])
print("ddarr[:,0] - ",ddarr[:,0])
print("ddarr[1,:] - ",ddarr[1,:])


# In[16]:


import numpy as np
d1=np.array([[1,2],
            [3,4],
            [5,6],
            [7,8],
            [9,10],
            [11,12],
            [13,14]])

print(d1[1: :2,1])
print(d1)
print(d1.shape)
print(d1.ndim)
print(d1[0: :2,0])


# In[18]:


import numpy as np
zr=np.zeros([3,4])
print("Zero filed array zr=\n",zr)
ar=np.ones([4,4])
print("1's filed array ones\n",ar)
print(ar*4)


# In[20]:


import numpy as np
ar=np.arange(1,6)
print("ar=",ar)
ar[3]=16
print("After updating,ar =",ar)


# In[22]:


import numpy as np
arr=np.array([11,22,33,0.,44,55])

print("arr.sum()=",arr.sum())
print(np.sum(arr))

print("arr.std()= ",arr.std())
print("arr.mean()= ",arr.mean())
print(np.mean(arr))

print("arr.max() =",arr.max())
print("arr.min() =",arr.min())

print("arr.size =",arr.size)
print("arr.shape =",arr.shape)


# In[23]:


import numpy as np
#Are all elements greater than 0
print(np.all([1,2,3,4]))
print(np.all([1,2,0,3,4]))
#Is any elements greater than zeroprint(np.any([1,2,3,4]))
print(np.any([1,2,0,3,4]))
print(np.any([0,0,0,0.,0]))
print(np.any([0,0,0,0.,0,1]))


# In[25]:


import numpy as np
n1=np.array([4,5,6])
n2=np.array([1,2,3])
print("n1= ",n1)
print("n2= ",n2)
print("n1 + n2 = ",n1+n2)
print("n1-n2=",n1-n2)

n3=np.array([4,6,7])
print(n1+n3)


# In[26]:


import numpy as np
n4=np.array([10,-1,0,90,300,3,-6,2])
print("Before:",n4)
n5=sorted(n4)
print(n5)
print("After: ",n4)
n4.sort()


# In[28]:


import numpy as np
n4=np.array([777,555,222,111,999,666])
print("n4=   ",n4)
print("n4.argsort():",n4.argsort())
indxArr=n4.argsort()
print("Min=" ,n4[indxArr[0]])
print("Max=",n4[indxArr[len(indxArr)-1]])
print("Sorted array=",n4[indxArr[:]])


# In[1]:


#27-12-22

import warnings
warnings.filterwarnings(action="ignore")
import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegresion


# In[1]:


#28-12-22

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import LogisticRegression
filename='indians-diabetes.data.csv'

hnames=['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe=pd.read_csv(filename,names=hnames)
array=dataframe.values
X=array[:,0:8]
Y=array[:,8]


model=LogisticRegression()
num_folds=10
kfold=KFold(n_splits=num_folds)
results=cross_val_score(model,X,Y,cv=kfold)
print("results: ",results)


# In[ ]:


import warnings
warnings.filterwarnings(action="ignore")
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename='indians-diabetes.data.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe=read_csv(filename,names=names)
array=dataframe.values
X=array[:,0:8]
Y=array[:,8]
loocv=LeaveOneOut()
model=LogisticRegression()
results=cross_val_score(model,X,Y,cv=loocv)
print("results: ",results)
print("result.size: ",results.size)
print("Sum of Positive Results: %i " %(results.sum()))


# In[ ]:


import warnings
warnings.filterwarnings(action="ignore")
from pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename='indians-diabetes.data.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe=read_csv(filename,names=names)
array=dataframe.values
X=array[:,0:8]
Y=array[:,8]
test_data_size=0.33
no_of_repetiions=4
shufflesplit=ShuffleSplit(n_splits=no_of_repetiions,test_size=test_data_size)
model=LogisticRegression()
results=cross_val_score(model,X,Y,cv=shufflesplit)
print(results)
print("Accuracy: %.3f" %(results.mean()*100.0))
print("Std.Deviation= %.3f" %(results.std()*100.0))

