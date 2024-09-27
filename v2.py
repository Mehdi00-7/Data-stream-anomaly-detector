import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



#I chose the Z-score algorithm as it measures how far a data point is away from the mean, if the zscore is greate than the threshhold, the data point is considered an anomaly
class AnomalyDetector:
    def __init__(self):
        self.alpha = 0.2 # 0.2 is a fair value for the mean to be responsive to recent values
        # the standard formula for alpha would be 2/(N+1) where N is the number of data points,
        # assuming we are testing with 1000 points alpha=2/(1000+1) whixh gives a really small mean less sensitive to recent changes
        self.threshold = 3 # zscore threshold for anomalies is usually set to 3
        self.mean = None  # set Running mean  to be none
        self.std = None  # set standard deviation to be initially none
        self.anomalies = []  # list to store anomalies (index, value)
    
    def update(self, value, index):
        if self.mean is None:  # First data point
            self.mean = value
            self.std = 1  # just to avoid division by 0
        else:
            # Update EWMA (mean) and standard deviation
            self.mean = self.alpha * value + (1 - self.alpha) * self.mean
            self.std = self.alpha * np.abs(value - self.mean) + (1 - self.alpha) * self.std

        # Compute zscore and detect anomalies
        z_score = (value - self.mean) / self.std #might cause division by 0, could have added epsilon to solve it
        if np.abs(z_score) > self.threshold:
            self.anomalies.append((index, value))
        
        return z_score

    def get_anomalies(self):
        return self.anomalies
def create_data_stream(detector, t, data, times, anomaly_points):
    # Generate a seasonal value using a sine wave
    seasonal_value = 10 * np.sin(2 * np.pi * t / 50)
    # Add some random noise to the data
    noise = np.random.normal(0, 1)
    # Introduce anomalies in the middle of the data stream
    if 100 < t < 200:
        anomaly = np.random.normal(30, 10)  # Higher chance of anomaly
    else:
        anomaly = 0  # No anomaly outside the middle range
    # Reset the detector's mean and std after the anomaly period to avoid skewing
    if t == 200:
        detector.mean = None
        detector.std = None

    # Combine the seasonal value, noise, and anomaly to get the final value
    value = seasonal_value + anomaly+ noise 

    # Update the anomaly detector with the new value
    z_score = detector.update(value, t)
    # Append the current time and value to their respective lists
    times.append(t)
    data.append(value)

    # If any anomalies are detected, add them to the anomaly_points list
    if detector.anomalies:
        anomaly_points.append((t, value))
    
    # Increment the time step
    return t + 1

def plot_data_stream():
    # Initialize the anomaly detector
    detector = AnomalyDetector()
    t = 0  # Start time
    data = []  # List to store data points
    times = []  # List to store time points
    anomaly_points = []  # List to store detected anomalies

    # Set up the plot
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='Data Stream')  # Line for the data stream
    anomaly_scatter = ax.scatter([], [], color='red', label='Anomalies')  # Scatter plot for anomalies
    ax.set_xlim(0, 100)  # Set x-axis limits
    ax.set_ylim(-15, 35)  # Set y-axis limits
    ax.set_title('Real-time Data Stream with Anomaly Detection')  # Set plot title
    ax.legend()  # Add legend

    def init():
        # Initialize the line and scatter plot with empty data
        line.set_data([], [])
        anomaly_scatter.set_offsets(np.empty((0, 2)))
        return line, anomaly_scatter

    def update(frame):
        nonlocal t
        # Generate the next data point and update the time
        t = create_data_stream(detector, t, data, times, anomaly_points)
        # Update the line plot with new data
        line.set_data(times, data)

        # Update the scatter plot with new anomalies
        if anomaly_points:
            anomaly_times, anomaly_values = zip(*anomaly_points)
            anomaly_scatter.set_offsets(np.c_[anomaly_times, anomaly_values])
        else:
            anomaly_scatter.set_offsets(np.empty((0, 2)))

        # Adjust the x and y axis limits dynamically
        ax.set_xlim(max(0, t - 100), t + 10)
        ax.set_ylim(min(data) - 5, max(data) + 5)
        
        return line, anomaly_scatter

    # Create the animation with 1000 frames
    ani = FuncAnimation(fig, update, init_func=init, blit=True, frames=1000, interval=100)
    plt.show()

# Start the data stream with visualization
plot_data_stream()