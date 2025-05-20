import plotly.graph_objects as go

import plotly.io as pio
pio.kaleido.scope.mathjax = None  # Disable MathJax rendering


# 1. Place in shelf
task_name = "Shelving an Item"
save_path = "place_in_shelf.svg"
x = ["Nav to<br>object", "Pick<br>object", "Nav to<br>shelf", "Place in<br>shelf"]
y1 = [1, 1, 1, 0.8]
y2 = [0.4, 0.4, 0.2, 0]
y3 = [0.4, 0.4, 0.2, 0]
y4 = [0.8, 0.8, 0.6, 0]
y5 = [1, 1, 1, 0.0]
y6 = [1, 1, 1, 0.2]
# marker_symbols = ["x", "x", "x", "circle"]
# marker_symbols = ["circle-open", "circle-open", "circle-open", "circle"]
# marker_sizes = [32, 32, 32, 20]

# # 2. Store Object in Drawer
# task_name = "Store in Drawer"
# save_path = "store_in_drawer.svg"
# x = ["Nav to<br>drawer", "Open<br>drawer", "Pick<br>object", "Place in<br>drawer", "Close<br>drawer"]
# y1 = [1, 0.8, 0.8, 0.8, 0.8]
# y2 = [0.6, 0.0, 0.0, 0.0, 0.0]
# y3 = [0.4, 0.0, 0.0, 0.0, 0.0]
# y4 = [1, 0.0, 0.0, 0.0, 0.0]
# y5 = [1, 0.4, 0.4, 0.0, 0.0]
# y6 = [1, 0.2, 0.2, 0.2, 0.2]
# marker_symbols = ["x", "circle", "x", "circle", "circle"]
# marker_symbols = ["circle-open", "circle", "circle-open", "circle", "circle"]
# marker_sizes = [32, 20, 32, 20, 20]

# # 3. Fill Water in Pot
# task_name = "Fill Pot"
# save_path = "fill_water.svg"
# x = ["Pick<br>pot", "Place pot<br>in sink", "Toggle on<br>faucet"]
# y1 = [1, 0.8, 0.6]
# y2 = [1, 0.0, 0.0]
# y3 = [1, 0.0, 0.0]
# y4 = [1, 0.0, 0.0]
# y5 = [1, 0.0, 0.0]
# y6 = [1, 0.2, 0.0]

# 4. Refrigerate
task_name = "Refrigerating an Item"
save_path = "refrigerate.svg"
x = ["Nav to<br>fridge", "Open<br>fridge", "Nav to<br>object", "Pick<br>object", "Nav with obj<br>to fridge", "Place in<br>fridge"]
y1 = [1, 1, 1, 1, 0.8, 0.6]
y2 = [0.6, 0.0, 0.0, 0.0, 0.0, 0.0]
y3 = [0.4, 0.0, 0.0, 0.0, 0.0, 0.0]
y4 = [0.8, 0.0, 0.0, 0.0, 0.0, 0.0]
y5 = [1, 0.0, 0.0, 0.0, 0.0, 0.0]
y6 = [1, 0.2, 0.2, 0.2, 0.2, 0.0]


# # 5. Place in box
# task_name = "Boxing an Item"
# save_path = "place_in_box.svg"
# x = ["Nav to<br>object", "Pick<br>object", "Place in<br>box"]
# y1 = [1, 1, 1]
# y2 = [0.6, 0.6, 0.2]
# y3 = [0.6, 0.6, 0.0]
# y4 = [1, 1, 0.2]
# y5 = [1, 1, 0.0]
# y6 = [1, 1, 0.4]

# # 6. Erase whiteboard
# task_name = "Erase Whiteboard"
# save_path = "erase_whiteboard.svg"
# x = ["Nav to<br>duster", "Pick<br>duster", "Erase<br>board"]
# y1 = [1, 1, 0.8]
# y2 = [0.8, 0.8, 0.0]
# y3 = [0.6, 0.6, 0.0]
# y4 = [0.8, 0.8, 0.2]
# y5 = [1, 1, 0.0]
# y6 = [1, 1, 0.0]

# # 7. Heat in oven
# task_name = "Load in Oven"
# save_path = "load_in_oven.svg"
# x = ["Open<br>oven", "Nav to<br>object", "Pick<br>object", "Nav to<br>oven", "Place in<br>oven", "Close<br>oven"]
# y1 = [1, 1, 1, 1, 0.8, 0.4]
# y2 = [0.2, 0.2, 0.2, 0.2, 0, 0]
# y3 = [0, 0, 0, 0, 0, 0]
# y4 = [0.2, 0.2, 0.2, 0.2, 0, 0]
# y5 = [0.4, 0.4, 0.4, 0.4, 0, 0]
# y6 = [0.4, 0.4, 0.4, 0.4, 0.2, 0.0]

graph_width = 200 * len(x)

# Create a line plot
fig = go.Figure()

# Add the first line
fig.add_trace(go.Scatter(
    x=x, y=y1,
    mode='lines+markers',
    name='SafeMimic',  # Legend entry
    line=dict(width=8, color="#228B22", dash="solid"),
    marker=dict(size=20), 
    opacity=0.8
))

# Add the second line
fig.add_trace(go.Scatter(
    x=x, y=y2,
    mode='lines+markers',
    name='Direct Exec (w/o SVF)',  # Legend entry
    line=dict(width=8, color="#D81B60", dash="longdash"),  # longdash
    marker=dict(size=20), 
    opacity=0.8
))

# Add the second line
fig.add_trace(go.Scatter(
    x=x, y=y3,
    mode='lines+markers',
    name='Direct Exec (w SVF)',  # Legend entry
    line=dict(width=8, color="#FF8C00", dash="dash"), # dash
    marker=dict(size=20), 
    opacity=0.8
))

# Add the second line
fig.add_trace(go.Scatter(
    x=x, y=y4,
    mode='lines+markers',
    name='Exploration (w/o SVF)',  # Legend entry
    line=dict(width=8, color="#800080", dash="dot"), #dot
    marker=dict(size=20), 
    opacity=0.8
))

# Add the second line
if y5 is not None:
    fig.add_trace(go.Scatter(
        x=x, y=y5,
        mode='lines+markers',
        name='IL (all safe actions)',  # Legend entry
        line=dict(width=8, color="#00008B", dash="dashdot"),  #dashdot
        marker=dict(size=20, symbol="circle"), 
        # marker=dict(size=marker_sizes, symbol=marker_symbols, line=dict(color="#1E88E5", width=2.5)), 
        opacity=0.8
))
    
# Add the second line
if y6 is not None:
    fig.add_trace(go.Scatter(
        x=x, y=y6,
        mode='lines+markers',
        name='IL (episode success actions)',  # Legend entry
        line=dict(width=8, color="#1E88E5", dash="longdashdot"),  #dashdot
        marker=dict(size=20, symbol="circle"), 
        # marker=dict(size=marker_sizes, symbol=marker_symbols, line=dict(color="#1E88E5", width=2.5)), 
        opacity=0.8
))



# Add titles and labels
fig.update_layout(
    title={
        'text': task_name,
        'x': 0.5,  # Position title at the center (x=0 is left, x=1 is right)
        'y': 0.90,  # Position title closer to the top (y=1 is top)
        # 'xanchor': 'center',  # Title alignment
        'yanchor': 'top',  # Align title to the top
        'font': dict(size=32, family='Times New Roman', color='black'),
    },
    xaxis=dict(
        # title="Segments",
        # title_font=dict(size=24, family='Times New Roman', color='black'),
        tickfont=dict(size=28, family='Times New Roman', color='black'),
        tickangle=0,  # Keep text horizontal
        tickmode="array",  # Use a custom list of ticks
        tickvals=x,  # Define the positions of the categories
        ticktext=x,  # Ensure each category label corresponds to a tick
        showgrid=True, gridcolor="lightgray",
        zeroline=True
    ),
    yaxis=dict(
        title="Success Rate",
        title_font=dict(size=32, family='Times New Roman', color='black'),
        tickfont=dict(size=28, family='Times New Roman', color='black'),
        ticklabelposition="outside",  # Ensure labels are outside the axis
        range=[-0.05, 1.05],  # Set the range for y-axis (min, max)
        showgrid=True, gridcolor="lightgray",
        zeroline=True,
        zerolinecolor="lightgray",  # Zero line color
        # zerolinewidth=2,  # Zero line thickness
    ),
    # plot_bgcolor='white',
    width=graph_width,
    # width=1400,
    height=600,
    # margin=dict(l=50, r=50, t=50, b=50),  # Compact margins
    legend=dict(
        orientation='h',  # Horizontal legend
        yanchor='bottom',  # Position legend below the top edge of the plot
        y=1.15,  # Adjust the vertical position of the legend
        xanchor='center',  # Center the legend horizontally
        x=0.5,  # Position legend in the center
        font=dict(
            size=16,  # Change font size of the legend text
            family="Times New Roman",  # Font family for legend
        )
    ),
    showlegend=False,  # Hide the legend
    plot_bgcolor="white",  # Plot background
    paper_bgcolor="white",  # Outer canvas background
)

# Show the plot
fig.show()
# fig.write_image(save_path, format='svg')
