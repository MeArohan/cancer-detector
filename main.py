import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def create_model(data): 
    X = data.drop(['diagnosis'],axis=1)  
    Y = data['diagnosis'] 

    

    #Normalizing/Scaling the predictor variables data (making it 0-1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)        

    #split the data into testing and training  

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= .2, random_state=42) 

    # training the model 

    model = LogisticRegression() 
    model.fit(X_train, Y_train)

    #test model (Compare results of what model predicted vs the actual results in the test data) 
    Y_pred = model.predict(X_test) 
    print('Accuracy of the model on testing', accuracy_score(Y_test, Y_pred)) 
    print("Classification Report: \n", classification_report(Y_test, Y_pred))

    return model, scaler


def get_clean_data():  
    data = pd.read_csv('cancer_data.csv') 
    data = data.drop(['Unnamed: 32', 'id'], axis=1)  
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def main(): 
    data = get_clean_data() 
    
    model,scaler = create_model(data)

    with open('model.pkl', 'wb') as f: 
        pickle.dump(model, f) 
    
    with open('scaler.pkl', 'wb') as f: 
        pickle.dump(scaler, f) 




if __name__ == '__main__': 
    main()
