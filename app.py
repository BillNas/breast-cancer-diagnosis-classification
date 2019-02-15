
import pandas as panda
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

kn_score=[]
tree_score=[]
logistic_score=[]
df=panda.read_csv('data.csv')
df.columns = df.columns.str.strip()
x=df.iloc[:,2:29]
y=df.iloc[:,0]

tree=DecisionTreeClassifier()
log=LogisticRegression()
kneighbors=KNeighborsClassifier(n_neighbors=2)
for i in range(0,4):
    xTrain,xTest,yTrain,yTest=tts(x,y,test_size=0.8)

    kneighbors.fit(xTrain,yTrain)
    log.fit(xTrain,yTrain)
    tree.fit(xTrain,yTrain)

    y1=kneighbors.predict(xTest)
    y2=log.predict(xTest)
    y3=tree.predict(xTest)

    kn_score.append(accuracy_score(yTest,y1))
    logistic_score.append(accuracy_score(yTest,y2))
    tree_score.append(accuracy_score(yTest,y3))

    plt.ylabel('Predicted Prognosis with KNeighbors Classifier')
    plt.scatter(xTest.iloc[:,1],y1,marker="|",color="red")
    plt.show()

    plt.ylabel('Predicted Prognosis with Logistic Regression Classifier')
    plt.scatter(xTest.iloc[:,1],y2,marker="|",color="red")
    plt.show()

    plt.ylabel('Predicted Prognosis with Tree Classifier')
    plt.scatter(xTest.iloc[:,1],y3,marker="|",color="red")
    plt.show()

kn_sum=sum(kn_score)
tree_sum=sum(tree_score)
logistic_sum=sum(logistic_score)


yBar=[(kn_sum/5*100),(tree_sum/5*100),(logistic_sum/5*100)]
xBar=['Mesos Oros KNeighbors','Mesos Oros Tree','Mesos Oros Logistic']

plt.bar(xBar, yBar, 1/1.15, color="blue")
plt.show()



