import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
from sklearn import tree
import graphviz
import seaborn as sns

url='/Purchase_Logistic.csv'
purchaseData = pd.read_csv(url)

X = purchaseData.iloc[:, [2, 3]]
Y = purchaseData.iloc[:, 4]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.25, random_state = 0) 

cf=DecisionTreeClassifier(random_state=23, max_depth=3);
cf.fit(Xtrain,Ytrain);
Ypred=cf.predict(Xtest)
cmat=confusion_matrix(Ytest,Ypred)

decPlot = plot_tree(decision_tree=cf, feature_names = ["Age", "Salary"], class_names =["No", "Yes"] , filled = True , precision = 4, rounded = True)

text_representation = tree.export_text(cf, feature_names = ["Age","Salary"])
print(text_representation)

print(cmat)
sns.heatmap(cmat,annot=True)


from sklearn.tree import export_graphviz
export_graphviz(cf, out_file="tree.dot", class_names=["No", "Yes"],feature_names=["Age", "Salary"], impurity=False, filled=True)
with open("tree.dot") as f:
    dot_graph = f.read()
graph=graphviz.Source(dot_graph)
graph.render(format='png',view='True')
graph.view()