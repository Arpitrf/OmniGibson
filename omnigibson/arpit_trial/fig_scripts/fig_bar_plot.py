import plotly.graph_objects as go
import numpy as np

# Sample data (replace with real values)
num_tasks = 7
num_methods = 6
success_rates = np.array([
    [0.2, 0.0, 0.2, 0.0, 0.4, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.2, 0.8],
    [0.0, 0.0, 0.2, 0.0, 0.0, 0.8],
    [0.0, 0.0, 0.0, 0.0, 0.2, 0.8],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.6],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.6],
])

# Replace 0s with a small nonzero bar height for visibility
MIN_BAR_HEIGHT = 0.02
success_rates = np.where(success_rates == 0, MIN_BAR_HEIGHT, success_rates)

tasks = ["Boxing an Item", "Shelving an Item", "Erase Whiteboard",
         "Store in Drawer", "Load in oven", "Refrigerating an Item", "Fill Pot"]
methods = [f"Method {i+1}" for i in range(num_methods)]

# Consistent colors per task
method_colors = ['#D81B60', '#FF8C00', '#800080', '#00008B', '#1E88E5', '#228B22']

# Create figure and add one bar trace per method
fig = go.Figure()

for method_idx in range(num_methods):
    fig.add_trace(
        go.Bar(
            x=tasks,
            y=success_rates[:, method_idx],
            name='',  # No legend
            marker_color=method_colors[method_idx],
            offsetgroup=method_idx,
            showlegend=False
        )
    )

# Layout adjustments for presentation
fig.update_layout(
    barmode='group',
    # title='Success Rate per Task',
    width=1600,  # Increased width
    height=600,
    xaxis=dict(
        # title='Tasks',
        tickangle=0,
        tickfont=dict(size=20),
        titlefont=dict(size=24),
        # showgrid=True,             # Show vertical grid lines
        # gridcolor='lightgray',     # Optional: custom grid color
        # gridwidth=1
    ),
    yaxis=dict(
        title='Success Rate',
        range=[0, 1.1],
        tickfont=dict(size=20),
        titlefont=dict(size=24),
        showgrid=True,             # Show horizontal grid lines
        gridcolor='lightgray',
        gridwidth=1
    ),
    font=dict(
        size=18,
        family='Avenir, sans-serif',  # Avenir with fallback
    ),
    plot_bgcolor='white',
    bargap=0.15,
)

fig.show()
save_path = "bar_plot.svg"
fig.write_image(save_path, format='svg')
