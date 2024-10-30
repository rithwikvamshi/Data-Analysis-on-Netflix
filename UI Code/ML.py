#!/usr/bin/env python
# coding: utf-8

# ## DATA

# In[1]:


import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from http.server import HTTPServer, BaseHTTPRequestHandler
import json


# ### DATA IMPORT

# In[2]:


final_df = pd.read_csv('final_data.csv')


# In[3]:


headings =final_df.columns.tolist()
print(headings)


# ### Model

# In[4]:


final_df = final_df.dropna(subset=['IMDb Score'])

final_df['genre'] = final_df['listed_in'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
final_df['Country Availability'] = final_df['Country Availability'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
final_df['Boxoffice'] = final_df['Boxoffice'].replace('[\$,]', '', regex=True).astype(float)

features = ['Hidden Gem Score','Boxoffice','rating','title','Writer','Languages','IMDb Votes','release_year','Production House','Awards Nominated For','Awards Received','type','listed_in','Country Availability', 'director', 'cast', 'country']
X = final_df[features]
y = final_df['IMDb Score']  

numeric_features = ['Hidden Gem Score','Boxoffice','release_year','IMDb Votes']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  
    ('scaler', StandardScaler())
])

categorical_features = ['rating','title','Writer','Languages','Production House','Awards Nominated For','Awards Received', 'type', 'Country Availability', 'director', 'cast', 'country']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
 
preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = RandomForestRegressor(n_estimators=100, random_state=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')
n = X_test.shape[0]  
p = X_test.shape[1]  
adjusted_r2 = 1 - (1-r2) * (n - 1) / (n - p - 1)
print(f'Adjusted R² Score: {adjusted_r2}')


# In[5]:


if final_df['duration'].dtype == object:
    final_df['duration'] = final_df['duration'].str.extract('(\d+)').astype(float)

final_df['Boxoffice'] = final_df['Boxoffice'].replace('Unknown', pd.NA)
final_df['Boxoffice'] = pd.to_numeric(final_df['Boxoffice'], errors='coerce')
final_df['Boxoffice'].fillna(final_df['Boxoffice'].median(), inplace=True)

# Calculate placeholders dynamically
numeric_features = ['Hidden Gem Score', 'IMDb Votes', 'Awards Received', 'Awards Nominated For', 'Boxoffice']
categorical_features = ['type', 'director', 'cast', 'country', 'release_year', 'rating', 'listed_in', 'Genre']

# Use median for numeric placeholders and mode for categorical placeholders
placeholders = {feature: final_df[feature].median() if final_df[feature].dtype in ['float64', 'int64'] else final_df[feature].mode()[0] for feature in numeric_features + categorical_features}

# Preparing features and target
selected_features = numeric_features + categorical_features
X = final_df[selected_features]
y = final_df['IMDb Score']

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# Regression pipeline
regression_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the regression model
regression_pipeline.fit(X_train, y_train)

# Predicting
y_pred = regression_pipeline.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)


# In[ ]:


# Function to dynamically compute placeholder values
def compute_dynamic_placeholders():
    # Assuming `final_df` is globally accessible or passed as a parameter
    numeric_features = ['Hidden Gem Score', 'IMDb Votes', 'Awards Received', 'Awards Nominated For', 'Boxoffice']
    categorical_features = ['cast', 'release_year', 'rating', 'listed_in']
    placeholders = {
        **{feature: final_df[feature].median() if final_df[feature].dtype in ['float64', 'int64'] else final_df[feature].mode()[0] for feature in numeric_features},
        **{feature: final_df[feature].mode()[0] for feature in categorical_features}
    }
    return placeholders

# Function to collect user input
def collect_user_input():
    print("Please enter the details for the prediction:")
    user_input = {
        'type': input("Enter type (Movie/TV Show): "),
        'director': input("Enter director: "),
        'country': input("Enter country: "),
        'Genre': input("Enter genre: ")  # Ensure this matches the training DataFrame column name
    }
    return user_input

# Function to predict the actual IMDb score using dynamic placeholder values
def predict_imdb_score_with_placeholders(data):
    user_input = {
        'type': data['type'],
        'director': data['director'],
        'country': data['country'],
        'Genre': data['genre']
    }
    # Compute placeholders dynamically
    placeholders = compute_dynamic_placeholders()
    
    # Update the user input with placeholder values for missing features
    full_input = {**placeholders, **user_input}  # Ensure dictionary keys exactly match your model's feature names
    input_df = pd.DataFrame([full_input])  # Let pandas handle the column ordering based on the keys of the full_input dictionary
    
    # Prediction using the trained regression pipeline
    predicted_score = regression_pipeline.predict(input_df)
    return {'prediction': predicted_score[0]}  # Return the predicted IMDb score

# Collecting user input
# user_input = collect_user_input()

# Example prediction call
# predicted_imdb_score = predict_imdb_score_with_placeholders(user_input)
# print(f"Predicted IMDb Score: {predicted_imdb_score:.2f}")  # Formatting to show up to two decimal places

# Rajiv Chilaka
# United States 
# Comedy


# Determining whether a Netflix title has a high IMDb score (defined as IMDb score >= 7.0) based on features such as 'Hidden Gem Score', 'IMDb Votes', 'Awards Received', 'Awards Nominated For', 'Boxoffice', 'type', 'director', 'cast', 'country', 'release_year', 'rating', and 'listed_in'. This model employed for this task is a Random Forest Classifier. The evaluation metrics used are Accuracy and Classification Report metrics (Precision, Recall, F1-score). The objective here is to develop a classification model that accurately predicts whether a Netflix title will have a high IMDb score or not.

# In[ ]:


def do_OPTIONS(self):
    self.send_response(200, "ok")
    self.send_header('Access-Control-Allow-Credentials', 'true')
    self.send_header('Access-Control-Allow-Origin', '*')
    self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    self.send_header('Access-Control-Allow-Headers', 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token')
    self.end_headers()

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def _set_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
    def do_OPTIONS(self):
        # Send allow headers for preflight requests
        self._set_headers()

    def do_GET(self):
        if self.path == '/':
            self.path = '/UI.html'
        # Serve your UI.html file
        try:
            file_to_open = open(self.path[1:]).read()
            self._set_headers()
            self.wfile.write(bytes(file_to_open, 'utf-8'))
        except:
            self._set_headers(404)
            self.wfile.write(bytes("File not found", 'utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        print("Received POST data:", body)  # Debugging: print received data

        data = json.loads(body)
        response = predict_imdb_score_with_placeholders(data)
        print("Response data:", response)  # Debugging: print response data

        response_data = json.dumps({'prediction': response})
        print("Sending response data:", response_data)  # Debugging: print response JSON
        self._set_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

    
if __name__ == '__main__':
    httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
    print("Server running")
    httpd.serve_forever()


# In[ ]:




