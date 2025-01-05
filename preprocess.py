import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('Cleaned.csv')
encode = LabelEncoder()
sc=StandardScaler()
def preprocessing_split(df):
    df = df.drop(df.columns[0], axis=1)
    df_categorical =[]
    for column in df.columns:
        if df[column].dtypes=="object":
            df_categorical.append(column)
    for column in df_categorical: 
        df[column] = encode.fit_transform(df[column])
    df = df.dropna(subset=['Financial Stress'])
    from sklearn.model_selection import train_test_split
    x = df.drop(columns=["Depression"])
    y =df["Depression"]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
    x_train=sc.fit_transform(x_train)
    x_test=sc.fit_transform(x_test)
    return x_train,x_test,y_train,y_test
