import numpy as np

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return np.array([w, x, y, z])

def rotate_vector_by_quaternion(vector, quaternion):
    # Convert vector to quaternion form
    v_quaternion = np.concatenate(([0], vector))

    # Calculate the inverse of the quaternion
    q_inverse = np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])

    # Rotate the vector using quaternion multiplication
    rotated_vector_quaternion = quaternion_multiply(quaternion, quaternion_multiply(v_quaternion, q_inverse))

    # Return the real part of the resulting quaternion as the rotated vector
    return rotated_vector_quaternion[1:]

def quaternion_to_rotation_matrix(quaternion):
    w, x, y, z = quaternion

    # Calculate the rotation matrix
    rotation_matrix = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])

    return rotation_matrix

def rotate_vector_by_matrix(vector, matrix):
    # Convert the vector to a column vector
    vector_column = np.array(vector).reshape(-1, 1)

    # Rotate the vector using matrix multiplication
    rotated_vector_column = np.dot(matrix, vector_column)

    # Return the resulting vector
    return rotated_vector_column.flatten()

# Define the rotation quaternion
rotation_quaternion = np.array([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0])

# Define the vector to be rotated
original_vector = np.array([2.0, 1.0, 3.0])

# Rotate the vector using quaternion
rotated_vector_quaternion = rotate_vector_by_quaternion(original_vector, rotation_quaternion)

# Convert the quaternion to a rotation matrix
rotation_matrix = quaternion_to_rotation_matrix(rotation_quaternion)

# Rotate the vector using the rotation matrix
rotated_vector_matrix = rotate_vector_by_matrix(original_vector, rotation_matrix)

# Print the results
print("Original Vector:", original_vector)
print("Rotated Vector (Quaternion):", rotated_vector_quaternion)
print("Rotated Vector (Matrix):", rotated_vector_matrix)
