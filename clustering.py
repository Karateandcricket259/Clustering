import pandas as pd
import plotly.express as px
reading=pd.read_csv('d00001.csv')
figure=px.scatter(reading,x='Size',y='Light')
figure.show()
print(reading.head())
from sklearn.cluster import KMeans
X=reading.iloc[:,[0,1]]
print(X)
wcss=[]
for i in range(1,11):
    kmeans1=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans1.fit(X)
    wcss.append(kmeans1.inertia_)
import matplotlib.pyplot as plt
import seaborn as sb 
plt.figure(figsize=(10,5))
sb.lineplot(range(1,11),wcss,marker='o',color='green')
plt.title('Points')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS Values')
plt.show()
kmeans1=KMeans(n_clusters=3,init='k-means++',random_state=42)
ymeans=kmeans1.fit_predict(X)
plt.figure(figsize=(15,7))
sb.scatterplot(X[ymeans==0,0],X[ymeans==0,1],label='Cluster 1',color='green')
sb.scatterplot(X[ymeans==1,0],X[ymeans==1,1],label='Cluster 2',color='orange')
sb.scatterplot(X[ymeans==2,0],X[ymeans==2,1],label='Cluster 3',color='blue')
sb.scatterplot(kmeans1.cluster_centers_[:,0],kmeans1.cluster_centers_[:,1],label='Center Point',color='red')
plt.grid(False)
plt.title('Cluster of Sizes and Amount of Light')
plt.xlabel('Size')
plt.ylabel('Light')
plt.legend()
plt.show() 
