import numpy as np
import matplotlib.pyplot as plt

# creating the dataset
data = {'Intensity_1': 0.18, 'Intensity_2': 0.65, 'Segment': 0.02,
        'Shock_Time_diff': 0.05, 'Usage_interval': 0.06, 'Type': 0.04}
courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(courses, values, color='green',
        width=0.4)

plt.xlabel("Features")
plt.ylabel("Importance Level")
plt.title("Per-Feature importance")
plt.show()