from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import pickle
import pandas as pd
from preprocess import preprocessing_split
from sklearn.ensemble import VotingClassifier
model=VotingClassifier(estimators=[
   ("logistic", LogisticRegression()),
    ("decision_tree", DecisionTreeClassifier(ccp_alpha=0.001)),
    ("random_forest", RandomForestClassifier()),
    ("gradient_boosting", GradientBoostingClassifier()),
    ("ada_boost", AdaBoostClassifier()),
    ("svc", SVC())
])
df = pd.read_csv("Cleaned.csv")
x_train, x_test, y_train, y_test = preprocessing_split(df)
model.fit(x_train,y_train)
with open(f"Ensemble.pkl", "wb") as file:
        pickle.dump(model, file)
