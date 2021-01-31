import matplotlib.pyplot as plt

# creating the dataset
data = {'Intensity_1': 0.18, 'Intensity_2': 0.65, 'Segment': 0.02,
        'Shock_Time_diff': 0.05, 'Usage_interval': 0.06, 'Type': 0.04}
features = list(data.keys())
values = list(data.values())

# Figure Size
fig, ax = plt.subplots(figsize=(16, 9))

# Horizontal Bar Plot
ax.barh(features, values)

# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']: ax.spines[s].set_visible(False)

# Remove x, y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x, y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)

# Show top values
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Per-Feature importance', loc='left', )

# Add Text watermark
fig.text(0.9, 0.15, 'Forkon', fontsize=10, color='grey', ha='right', va='bottom', alpha=0.7)

# Show Plot
plt.show()