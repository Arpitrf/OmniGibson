import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def sample_from_cone_final(original_vector, max_angle=np.pi/18, num_samples=10, norm_variance=0.4):
    original_vector = original_vector / np.linalg.norm(original_vector)  # Normalize input vector
    original_norm = np.linalg.norm(original_vector)
    noisy_vectors = []

    # Compute the rotation matrix to align [0, 0, 1] with the original vector
    z_axis = np.array([0.0, 0.0, 1.0])
    if np.allclose(original_vector, z_axis):
        rotation_matrix = np.eye(3)  # No rotation needed if already aligned
    else:
        rotation_axis = np.cross(z_axis, original_vector)
        rotation_axis /= np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(z_axis, original_vector), -1.0, 1.0))
        rotation_matrix = R.from_rotvec(angle * rotation_axis).as_matrix()

    for _ in range(num_samples):
        # Sample a random point in the cone aligned with the z-axis
        z = np.cos(max_angle) + (1 - np.cos(max_angle)) * np.random.rand()
        phi = 2 * np.pi * np.random.rand()
        x = np.sqrt(1 - z**2) * np.cos(phi)
        y = np.sqrt(1 - z**2) * np.sin(phi)
        random_point = np.array([x, y, z], dtype=np.float64)

        # Apply the rotation to align the point with the original vector
        noisy_vector = rotation_matrix @ random_point

        # Vary the norm of the noisy vector
        varied_norm = original_norm * (1 + np.random.uniform(-norm_variance, norm_variance))
        noisy_vector *= varied_norm

        print("np.linalg.norm(noisy_vector): ", np.linalg.norm(noisy_vector))
        noisy_vectors.append(noisy_vector)

    return np.array(noisy_vectors)


np.random.seed(0)
original_vector = np.array([1.0, 0.0, 1.0])
print("np.linalg.norm(original_vector): ", np.linalg.norm(original_vector))
sampled_vectors = []

sampled_vectors = sample_from_cone_final(original_vector, max_angle=np.pi/8)

# for _ in range(5):
#     sampled_vector = sample_from_cone(original_vector, max_angle=np.pi/18)
#     sampled_vectors.append(sampled_vector)


# show the original vector and the noisy vector in matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')    
ax.quiver(0, 0, 0, original_vector[0], original_vector[1], original_vector[2], color='r')
for sampled_vector in sampled_vectors:
    ax.quiver(0, 0, 0, sampled_vector[0], sampled_vector[1], sampled_vector[2], color='b')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1])
plt.show()