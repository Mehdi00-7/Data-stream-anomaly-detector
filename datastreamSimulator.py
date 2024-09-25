import numpy as np
import matplotlib.pyplot as plt
#could have imported zscore directly and used it from scipy.stats import zscore



# Function to generate a stream of random data points
def generateStream(num_points):
    #to ensure reproducibility
    np.random.seed(42)
    #array of time point
    time = np.arange(num_points)
    #set seasonal pattern to be repeated using sine wave
    seasonal_pattern = 5 * np.sin(2 * np.pi * time / 50)
    #set random noise
    noise = np.random.normal(0, 1, num_points)
    #set increasing trend over time
    trend = 0.05 * time
    #create data stream
    data_stream = seasonal_pattern + noise + trend
    #create points with different mean ,10% of total points
    anomPoints = np.random.normal(10, 1, (int(num_points*0.1)))
    data_stream[num_points-int(num_points*0.1):] = anomPoints

    return data_stream



#1-detect anomalies using zscore
def detectanom(data):
    #set threshhold to be 1.5
    threshold =3
    #calculates the mean
    mean = np.mean(data)
    #calculates the standard derivation
    sdv = np.std(data)
    #create an empty list to store anomalies
    anom = []
    #iterate over each value in dataset
    for i,x in enumerate(data):
        #calculate its z score
        z_score = abs((x - mean) / sdv)
        #if zscore greater than the value, consider the value as anomaly and add its index to the list
        if z_score > threshold:
            anom.append(i)
    #return the list of anomalies found in the dataset
    return anom

#2-detect anàmalies using interquantile range
def detectanom2(data):
    anom=[]
    mydata=data.copy()
    quantile1,quantile3=np.percentile(mydata,[25,75])
    Iqr_val=quantile3-quantile1
    min_val=quantile1-(1.5*Iqr_val)
    max_val=quantile3+(1.5*Iqr_val)
    for i,x in enumerate(data):
        if x<min_val or x>max_val:
            anom.append(i)
    return anom

def visualize_data_stream(data, anomalies):
    #create figure size
    plt.figure(figsize=(10,6))
    #plot the data stream
    plt.plot(data,color='black',label='Data Stream')
    #highlight the anomalies
    plt.scatter(anomalies, data[anomalies], color='red', label='Anomalies')
    #add a legend
    plt.legend()
    plt.title('Data Stream with Detected Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()
#generate stream with 100 poinnts
data= generateStream(100)
#data = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(10, 1, 10)])
#detect anomalies
anomalies = detectanom2(data)
#visualise the stream
visualize_data_stream(data, anomalies)

