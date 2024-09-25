from dv import NetworkEventInput
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from sklearn.cluster import DBSCAN


#initilaize event data arrays
x = []
y = []
polarity = []
timestamp = []


# Initialize Matplotlib figure and scatter plot
fig, ax = plt.subplots()
scatter = ax.scatter([], [], c=[], cmap='rainbow')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Real-time Event Clustering')


# Clustering parameters
k = 3  # Number of clusters
dbscan = DBSCAN(eps = ??, min_samples= ???)

# Update function for animation
def update(frame):
    global x, y, polarity, timestamp

    # Clear event data arrays
    x = []
    y = []
    polarity = []
    timestamp = []

    # Read events from the live camera stream
    # You need to implement this part based on your camera interface

    # Process the new events
    data = np.column_stack((x, y))
    
    labels = dbscan.fit_predict(data)

    # Update scatter plot data
    scatter.set_offsets(np.column_stack((x, y)))
    scatter.set_array(labels)

# Set up the animation
ani = FuncAnimation(fig, update, interval=100)  # Adjust the interval as needed

# Show the plot
plt.show()




