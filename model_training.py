# from sklearn.linear_model import LogisticRegression
# import pickle
# import pandas as pd
# from preprocess import preprocessing_split
# model=LogisticRegression()
# df=pd.read_csv("Cleaned.csv")
# x_train,x_test,y_train,y_test=preprocessing_split(df)
# model.fit(x_train,y_train)
# with open("logisticpkl.pkl","wb") as file:
#     pickle.dump(model,file)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import pickle
import pandas as pd
from preprocess import preprocessing_split

# Load the dataset
df = pd.read_csv("Cleaned.csv")
x_train, x_test, y_train, y_test = preprocessing_split(df)

# List of models to train and pickle
model_list = [
    ("LogisticRegression", LogisticRegression()),
    ("DecisionTreeClassifier", DecisionTreeClassifier(ccp_alpha=0.001)),
    ("RandomForestClassifier", RandomForestClassifier()),
    ("GradientBoostingClassifier", GradientBoostingClassifier()),
    ("AdaBoostClassifier", AdaBoostClassifier()),
    ("SVC", SVC())
]

# Train and pickle each model
for model_name, model in model_list:
    model.fit(x_train, y_train)
    
    # Save the model using pickle
    with open(f"{model_name}.pkl", "wb") as file:
        pickle.dump(model, file)
    print(f"{model_name} saved as {model_name}.pkl.")










