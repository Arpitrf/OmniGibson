import open3d as o3d
import numpy as np

import grasp_pose_generator as gpg
from grasp_selector import GraspSelector

def main():
    num_samples = 500
    test_cloud_file = "/home/arpit/projects/OmniGibson/DoorCIP.ply"
    test_cloud_with_normals = o3d.io.read_point_cloud(test_cloud_file)
    world_frame_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()

    door_frame = np.eye(4)
    door_frame[:3, :3] = np.array([\
                                    [0, 0, 1], \
                                    [0, 1, 0], \
                                    [-1, 0, 0]])
    door_frame[:3, 3] = [0.14, 0.348, 0.415]

    gs = GraspSelector(door_frame, test_cloud_with_normals)

    sampled_poses = gs.getRankedGraspPoses()
    print("sampled_poses: ", np.array(sampled_poses).shape)
    desired_sampled_poses = sampled_poses[:num_samples]
    desired_sampled_poses = [gpg.translateFrameNegativeZ(p, gs.dist_from_point_to_ee_link) for p in desired_sampled_poses]

    gs.visualizeGraspPoses(desired_sampled_poses)


if __name__ == '__main__':
    main()