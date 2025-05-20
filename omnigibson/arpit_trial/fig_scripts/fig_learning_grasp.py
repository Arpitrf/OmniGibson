import plotly.graph_objects as go


# 1. Place in shelf
x = ["SafeMimic", "SafeMimic +<br>Policy Memory"]
y1 = [76, 30]
y2 = [70, 23]
y3 = [72, 41]
y4 = [111, 46]

# Create a line plot
fig = go.Figure()

# Add the first line
fig.add_trace(go.Scatter(
    x=x, y=y1,
    mode='lines+markers',
    name='Shelving item',  # Legend entry
    line=dict(width=8, color="#228B22", dash="solid"),
    marker=dict(size=16), 
    opacity=0.8
))

# Add the second line
fig.add_trace(go.Scatter(
    x=x, y=y2,
    mode='lines+markers',
    name='Boxing item',  # Legend entry
    line=dict(width=8, color="#00008B", dash="solid"),  
    marker=dict(size=16), 
    opacity=0.8
))

# Add the second line
fig.add_trace(go.Scatter(
    x=x, y=y3,
    mode='lines+markers',
    name='Store in Drawer',  # Legend entry
    line=dict(width=8, color="#FF8C00", dash="solid"), 
    marker=dict(size=16), 
    opacity=0.8
))

# Add the second line
fig.add_trace(go.Scatter(
    x=x, y=y4,
    mode='lines+markers',
    name='Refrigerating item',  # Legend entry
    line=dict(width=8, color="#800080", dash="solid"), 
    marker=dict(size=16), 
    opacity=0.8
))


# Add titles and labels
fig.update_layout(
    title={
        # 'text': task_name,
        # 'x': 0.5,  # Position title at the center (x=0 is left, x=1 is right)
        # 'y': 0.9,  # Position title closer to the top (y=1 is top)
        # 'xanchor': 'center',  # Title alignment
        # 'yanchor': 'top',  # Align title to the top
        # 'font': dict(size=32, family='Times New Roman', color='black'),
    },
    xaxis=dict(
        # title="Segments",
        # title_font=dict(size=24, family='Times New Roman', color='black'),
        tickfont=dict(size=24, family='Times New Roman', color='black'),
        tickangle=0,  # Keep text horizontal
        tickmode="array",  # Use a custom list of ticks
        tickvals=x,  # Define the positions of the categories
        ticktext=x,  # Ensure each category label corresponds to a tick
        showgrid=True, gridcolor="lightgray",
        zeroline=True
    ),
    yaxis=dict(
        title="# Waypoints Explored",
        title_font=dict(size=28, family='Times New Roman', color='black'),
        tickfont=dict(size=24, family='Times New Roman', color='black'),
        ticklabelposition="outside",  # Ensure labels are outside the axis
        range=[-2, 120],  # Set the range for y-axis (min, max)
        showgrid=True, gridcolor="lightgray",
        zeroline=True,
        zerolinecolor="lightgray",  # Zero line color
        # zerolinewidth=2,  # Zero line thickness
    ),
    # plot_bgcolor='white',
    # width=graph_width,
    width=600,
    height=600,
    # margin=dict(l=50, r=50, t=50, b=50),  # Compact margins
    legend=dict(
        orientation='h',  # Horizontal legend
        yanchor='bottom',  # Position legend below the top edge of the plot
        y=1.05,  # Adjust the vertical position of the legend
        xanchor='center',  # Center the legend horizontally
        x=0.5,  # Position legend in the center
        font=dict(
            size=16,  # Change font size of the legend text
            family="Times New Roman",  # Font family for legend
        )
    ),
    showlegend=True,  # Hide the legend
    plot_bgcolor="white",  # Plot background
    paper_bgcolor="white",  # Outer canvas background
)

# Show the plot
# fig.show()
fig.write_image("policy_memory.svg", format='svg')
