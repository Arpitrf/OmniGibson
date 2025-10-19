import plotly.graph_objects as go

# Data
methods = ["SafeMimic", "SafeMimic + <br>Policy Memory"]
values = [83, 35]  # You can replace these with your actual values

# Create figure
fig = go.Figure(data=[
    go.Bar(x=methods, y=values, marker_color=["#C9DAF8", "#FFE7A7"], width=0.5)  # Use clean, distinct colors
])

# Update layout
fig.update_layout(
    title="",  # Leave title empty for presentation slides
    width=900,
    height=900,
    font=dict(family="Avenir", size=40),
    plot_bgcolor='white',
    xaxis=dict(showline=True, linewidth=1, linecolor='black'),
    yaxis=dict(range=[0, 110], title="# Waypoints Explored", showline=True, linewidth=1, linecolor='black', gridcolor='lightgray'),
    margin=dict(l=40, r=20, t=20, b=40)
)

fig.show()
save_path = "bar_plot_policy_memory.svg"
fig.write_image(save_path, format='svg')
