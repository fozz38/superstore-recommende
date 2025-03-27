from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as  np
import joblib
import os

np._core = np.core

app = Flask(__name__)

# Load the pre-trained models
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

df = pd.read_csv('Superstore Sales Dataset.csv')

# Preprocessing
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.lower().str.strip()

df['Order Date'] = pd.to_datetime( df['Order Date'], dayfirst=True, yearfirst=False, errors='coerce')
df['Ship Date'] = pd.to_datetime( df['Ship Date'], dayfirst=True, yearfirst=False, errors='coerce')

df['Postal Code'] = df['Postal Code'].astype(str)
df['Postal Code'] = df['Postal Code'].fillna("05408")
df.drop(columns=['Row ID'], inplace=True)
df = df.drop_duplicates()

# Customer segmentation 
customer_sales = df.groupby('Customer ID', as_index=False)['Sales'].median()
customer_sales['sales_scaled'] = scaler.transform(customer_sales[['Sales']])
customer_sales['Cluster'] = kmeans.predict(customer_sales[['sales_scaled']])

df = df.merge(customer_sales[['Customer ID', 'Cluster']], on='Customer ID', how='left')

def recommend_products(cluster):
    # Recommend top 10 products for a given cluster
    cluster_products = df[df['Cluster'] == cluster]
    top_products = cluster_products.groupby('Product Name')['Sales'].sum().nlargest(10).index.tolist()
    return top_products if top_products else ["No recommendations available"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    customer_id = request.form['customer_id'].lower().strip()
    
    # Check if customer exists
    if customer_id not in df['Customer ID'].unique():
        return render_template('error.html')
    
    # Get cluster for the customer
    cluster = customer_sales[customer_sales['Customer ID'] == customer_id]['Cluster'].values[0]
    
    # Get recommendations
    recommendations = recommend_products(cluster)
    
    return render_template('results.html', 
                         customer_id=customer_id,
                         recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)