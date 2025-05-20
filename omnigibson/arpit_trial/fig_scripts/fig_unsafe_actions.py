import plotly.graph_objects as go

x = ["Direct Exec<br>(w/o SVF)", "Direct Exec<br>(with SVF)", "Exploration<br>(w/o SVF)", "IL (all safe actions)", "IL (episode success actions)", "SafeMimic"]
y1 = [13.4, 0.5, 14.2, 10.8, 9.5, 0.5]
colors = ["#D81B60", "#FF8C00", "#800080", "#00008B", "#1E88E5", "#228B22"]

# Create a line plot
fig = go.Figure()

# Add the first line
fig.add_trace(go.Bar(
    x=x, y=y1,
    marker=dict(
        color=colors,
        line=dict(width=1.5)  # Border for bars
    ),
    opacity=0.8
))


# Add titles and labels
fig.update_layout(
    title={
        # 'text': task_name,
        # 'x': 0.5,  # Position title at the center (x=0 is left, x=1 is right)
        # 'y': 0.9,  # Position title closer to the top (y=1 is top)
        'xanchor': 'center',  # Title alignment
        # 'yanchor': 'top',  # Align title to the top
        'font': dict(size=32, family='Avenir', color='black'),
    },
    xaxis=dict(
        # title="Segments",
        # title_font=dict(size=24, family='Avenir', color='black'),
        tickfont=dict(size=26, family='Avenir', color='black'),
        tickangle=0,  # Keep text horizontal
        tickmode="array",  # Use a custom list of ticks
        tickvals=x,  # Define the positions of the categories
        ticktext=x,  # Ensure each category label corresponds to a tick
        # showgrid=True, gridcolor="lightgray",
        zeroline=True
    ),
    yaxis=dict(
        title="Unsafe Actions (%)",
        title_font=dict(size=32, family='Avenir', color='black'),
        tickfont=dict(size=26, family='Avenir', color='black'),
        ticklabelposition="outside",  # Ensure labels are outside the axis
        range=[0.0, 100.5],  # Set the range for y-axis (min, max)
        showgrid=True, gridcolor="lightgray",
        zeroline=True,
        zerolinecolor="lightgray",  # Zero line color
        # zerolinewidth=2,  # Zero line thickness
    ),
    # plot_bgcolor='white',
    # width=graph_width,
    width=900,
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
            family="Avenir",  # Font family for legend
        )
    ),
    showlegend=False,  # Hide the legend
    plot_bgcolor="white",  # Plot background
    paper_bgcolor="white",  # Outer canvas background
)

# Show the plot
fig.show()
# fig.write_image("unsafe_actions_plot.svg", format='svg')
