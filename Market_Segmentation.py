#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 00:15:12 2020

@author: chenxidong
"""

import os
os.chdir('/Users/chenxidong/Desktop/6004 Market Segmentation/')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # drawing graphs
from sklearn.tree import DecisionTreeClassifier # a classification tree
from sklearn.tree import plot_tree # draw a classification tree
from sklearn.model_selection import cross_val_score # cross validation
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 


import seaborn as sns
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def main():
    st.title('💳 Credit Card User Segmentation')
    st.sidebar.title('💳 EDA and Clustering control panel')
    
    st.markdown('User Segmentation by K-Means Clustering and PCA Visualization - SDSC6004 Project in 2020')
    st.markdown('Data source: https://www.kaggle.com/arjunbhasin2013/ccdata')
    st.sidebar.markdown('Creater: DONG CHENXI')
    #simpply use the cache last time unlesss the input changed

    def load_data():
        data=pd.read_csv('/Users/chenxidong/Desktop/6004 Market Segmentation/marketing_data.csv')
        # Fill up the missing elements with mean of the 'MINIMUM_PAYMENT' 
        data.loc[(data['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].mean()
        # Fill up the missing elements with mean of the 'CREDIT_LIMIT' 
        data.loc[(data['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] =data['CREDIT_LIMIT'].mean()
        #drop the cust_id col
        data.drop('CUST_ID',axis=1,inplace=True)

        return data
    
 
    
    #plot the evaluation metrics
    def plot_metrics(metrics_list):
        # Confusion Matrix
        if 'Cluster Histogram' in metrics_list:
            st.subheader('Cluster Histogram')
            # Plot the histogram of various clusters
            for i in df.columns:
                plt.figure(figsize = (35, 5))
                for j in range(num_of_clusters):
                    plt.subplot(1,num_of_clusters,j+1)
                    cluster = df_cluster[df_cluster['cluster'] == j]
                    cluster[i].hist(bins = 20)
                    plt.title('{}    \nCluster {} '.format(i,j))
                plt.show()
                st.pyplot()
        
           
            
    df=load_data()
    
    #explore the data before create the model
    st.sidebar.subheader("Explore Data Analysis")
    
    if st.sidebar.checkbox('Show Attribute Infomation',False):
        st.subheader('Attribute Info')
        from PIL import Image
        Attribute_info= Image.open('data dictionary.png')
        st.image(Attribute_info,use_column_width=True)
        
    if st.sidebar.checkbox('Display dataset',False):
        st.subheader('Dataset')
        st.write(df)
      
    if st.sidebar.checkbox('Show Data description',False):
        st.subheader('Dataset description')
        st.write(df.describe())
        
    if st.sidebar.checkbox('Show Attributes Histogram',False):
        st.subheader('Attributes Histogram')
        from PIL import Image
        hist= Image.open('attribute_hist.png')
        st.image(hist,use_column_width=True)
        
    if st.sidebar.checkbox('Attributes correlation heat map',False):
        st.subheader('Attributes correlation heatmap')
        from PIL import Image
        heatmap= Image.open('heatmap.png')
        st.image(heatmap,use_column_width=True)
    
    if st.sidebar.checkbox('Optimal number of clusters',False):
        st.subheader('Elbow method: WCSS-num of cluster plot')
        from PIL import Image
        elbow= Image.open('elbow.png')
        elbow_plot= Image.open('elbow_plot.png')
        st.image(elbow,use_column_width=True)
        st.image(elbow_plot,use_column_width=True)
    
    
    #create the model
    st.sidebar.subheader('Clustering and Visualization')
    classifier=st.sidebar.selectbox('Model',('K-Means Clustering','PCA Visualization'))
    
    #K-Means clustering
    if classifier == 'K-Means Clustering':
        st.sidebar.subheader('Model Parameters')
        #cluster number
        cluster_num=st.sidebar.number_input('Cluster Number',4,10,step=1,key='cluster_num')
        # Let's scale the data first
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)

    

        
        if st.sidebar.button('Run',key='classify'):
            st.subheader('K-means cluster results')
            num_of_clusters=cluster_num
            model = KMeans(num_of_clusters,random_state=0)
            model.fit(df_scaled)
            labels = model.labels_
            df_cluster = pd.concat([df, pd.DataFrame({'cluster':labels})], axis = 1)
            
            metrics='Cluster Histogram'
            plot_metrics(metrics)
            #display the optimal tree diagram
            from PIL import Image
            pie_chart = Image.open('pie_app.png')
            st.image(pie_chart, caption='An example of 7-clusters user distribution',use_column_width=True)
           
            
            
    #PCA model        
    if classifier == 'PCA Visualization':
        st.sidebar.subheader('Model Parameters: Set the cluster number equal to the K-Means model')
        #cluster number
        cluster_num=st.sidebar.number_input('Cluster Number',4,10,step=1,key='cluster_num')
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        
        if st.sidebar.button('Run',key='classify'):
            st.subheader('2D PCA visualization for user clustering')
            num_of_clusters=cluster_num
            model_k= KMeans(num_of_clusters)
            model_k.fit(df_scaled)
            labels = model_k.labels_
            model=PCA(n_components=2)
            principal_comp=model.fit_transform(df_scaled)
            pca_df = pd.DataFrame(data = principal_comp, columns =['pca1','pca2'])
            pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)
            #plot
            plt.figure(figsize=(7,7))
            ax = sns.scatterplot(x="pca1", y="pca2",hue = "cluster", data = pca_df,palette="deep")
            plt.show()
            st.pyplot()
    
    

if __name__ == '__main__':
    main()

st.set_option('deprecation.showPyplotGlobalUse', False)
 

