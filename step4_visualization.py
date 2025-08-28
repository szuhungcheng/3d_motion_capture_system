import os
import pandas as pd
import plotly.graph_objects as go

# --- SETTINGS ---
csv_dir = 'output_3d_data'
X_LIMITS = [-3000, 3000]
Y_LIMITS = [-3000, 3000]
Z_LIMITS = [-1000, 5000]

# --- LOOP THROUGH ALL CSV FILES ---
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
csv_files.sort()

for file_name in csv_files:
    file_path = os.path.join(csv_dir, file_name)
    print(f"\n📈 Visualizing: {file_name}")

    # Load CSV
    df = pd.read_csv(file_path)

    # Frame and joint parsing
    frames = df['Frame'].values
    keypoints = [col for col in df.columns if col != 'Frame']
    num_joints = len(keypoints) // 3
    joint_names = [keypoints[i * 3].split('_')[0] for i in range(num_joints)]

    # Create figure
    fig = go.Figure()

    # Animation frames
    frames_list = []
    for frame_idx in range(len(frames)):
        frame_data = []
        for i in range(num_joints):
            x = [df.iloc[frame_idx, i * 3 + 1]]
            y = [df.iloc[frame_idx, i * 3 + 2]]
            z = [df.iloc[frame_idx, i * 3 + 3]]

            frame_data.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=5),
                name=joint_names[i],
                showlegend=False
            ))

        frames_list.append(go.Frame(data=frame_data, name=str(frame_idx)))

    # Add first frame
    for i in range(num_joints):
        x = [df.iloc[0, i * 3 + 1]]
        y = [df.iloc[0, i * 3 + 2]]
        z = [df.iloc[0, i * 3 + 3]]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=5),
            name=joint_names[i]
        ))

    # Layout
    fig.update_layout(
        title=f"3D Joint Animation: {file_name}",
        scene=dict(
            xaxis=dict(title='X', range=X_LIMITS),
            yaxis=dict(title='Y', range=Y_LIMITS),
            zaxis=dict(title='Z', range=Z_LIMITS),
            aspectmode='cube'
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(label='Play',
                         method='animate',
                         args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                ]
            )
        ],
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.frames = frames_list

    # Show the animation
    fig.show()

    # Ask user whether to continue
    response = input("\n▶️ Press Enter to view next file, or type 'q' to quit: ").strip().lower()
    if response == 'q':
        print("🛑 Visualization stopped.")
        break
