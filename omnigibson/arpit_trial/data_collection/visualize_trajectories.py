import pickle
import cv2
import h5py

import numpy as np
np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def show_vectors(hdf5_file):
    # show the original vector and the noisy vector in matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    # ax.quiver(0, 0, 0, original_vector[0], original_vector[1], original_vector[2], color='r')
    for i, ep in enumerate(hdf5_file["data"]):
        # if i != 7:
        #     continue
        vector = np.array(hdf5_file[f"data/{ep}/actions/complete_actions"])[0]
        grasp_vector = np.array(hdf5_file[f"data/{ep}/extras/grasps"])
        print("Episode: ", ep)
        print("vector: ", vector)
        print("grasp_vector: ", grasp_vector)
        print("==============")
        color = "g"
        if any(grasp_vector == False):
            color = "r"
        ax.text(vector[0], vector[1], vector[2], f"{i}", color=color)
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color=color)
        # breakpoint()
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.2, 0.2])
    plt.show()

def visualize_trajectories(actions, start_position, ax=None, color='r', ep=0, grasp_vector=None, base_orn=None): 
    total_lines = 0
    for i in range(actions.shape[0]):
        trajectory = actions[i, :, 3:6]
        prev_position = start_position

        for j in range(trajectory.shape[0]):
            direction = trajectory[j]  # Direction vector at this waypoint
            # convert delta from robot frame to world frame
            if base_orn is not None:
                base_orn_matrix = R.from_quat(base_orn).as_matrix()
                direction = base_orn_matrix @ direction
            magnitude = np.linalg.norm(direction)  # Magnitude of the direction vector
            direction_normalized = direction / magnitude if magnitude != 0 else direction  # Normalize the direction
            step = magnitude * direction_normalized

            next_position = prev_position + step
            if grasp_vector is not None:
                if grasp_vector[j+1]:
                    color = "g"
                else:
                    color = "r"
            ax.quiver(prev_position[0], prev_position[1], prev_position[2], step[0], step[1], step[2], color=color)
            # ax.quiver(0, 0, 0, step[0], step[1], step[2], color=color)
            # visualize_marker(start_position=prev_position, end_position=next_position, id=total_lines)
            prev_position = prev_position + step  # Move to the new position
            total_lines += 1

        # ax.text(next_position[0], next_position[1], next_position[2], f"{ep}", color=color)

    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([-0.2, 0.2])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')


def main():
    # change the filename here
    f = h5py.File("/home/arpit/projects/OmniGibson/open_cabinet/dataset.hdf5", "r")
    print(len(f["data"].keys()))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(f["data"])):
        # remove this in case not needed
        if i > 10:
            break
        print("Episode: ", i)
        print("len(actions), len(obs): ", np.array(f[f"data/episode_{i:05d}/actions/actions"]).shape, np.array(f[f"data/episode_{i:05d}/observations/rgb"]).shape)
        print("grasps: ", np.array(f[f"data/episode_{i:05d}/extras/grasps"]))

        if len(np.array(f[f"data/episode_{i:05d}/actions/actions"])) == 0:
            continue
        # print("actions: ",  np.array(f[f"data/episode_{i:05d}/actions/actions"])[:, 3:6])
        action_traj = np.array(f[f"data/episode_{i:05d}/actions/actions"])
        grasp_vector = np.array(f[f"data/episode_{i:05d}/extras/grasp_label"])
        base_orn = np.array(f[f"data/episode_{i:05d}/proprioceptions/base_orn"])[0]
        visualize_trajectories(action_traj[None, ...], np.array([0.0, 0.0, 0.0]), ax=ax, ep=i, grasp_vector=grasp_vector, base_orn=base_orn)
    plt.show()

if __name__ == "__main__":
    main()
