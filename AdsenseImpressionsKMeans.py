import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

#Predict most likely position of impression using KMeans algorithm,
#Abs Top is probability that impression is in position number 1
#Top is the probability that impression is in one of the top 3 positions.

#Read in AdSense Impression Data
data = pd.read_csv("AdsenseImpressions.csv")

#drop missing values (--)
index2Drop = data[data['Impr. (Abs. Top) %']==" --"].index
index2Drop2 = data[data['Impr. (Top) %']==" --"].index
data = data.drop(index=index2Drop)

#Keep only the two features needed
absTop = data['Impr. (Abs. Top) %']
Top = data['Impr. (Top) %']
keywords = data['Keyword']

#convert to decimal values
absTop = absTop.str.rstrip('%')
absTop = pd.to_numeric(absTop)
Top = Top.str.rstrip('%')
Top = pd.to_numeric(Top)

X = np.array(absTop,Top)
X = X.reshape(-1,1)
#Nature of problem sets number of clusters to 3
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
labels = kmeans.labels_

#Create Data to export
dataPredicted = pd.DataFrame({'Keyword':keywords,'absTop':absTop,'Top':Top,'PredictedGroup':labels})
#Visualize predicted labels
plt.scatter(x=dataPredicted.absTop, y=dataPredicted.Top,c=dataPredicted.PredictedGroup)
plt.xlabel("Absolute Top")
plt.ylabel("Top")
plt.show()

#Write data to path (insert path of your chooseing)
dataPredicted.to_csv("....csv")

print(kmeans.labels_)
print(data.head())

