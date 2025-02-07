import numpy as np
import quaternion
from racer_datagen.data_aug.base import DataAugmentor

    
augmentor = DataAugmentor(task_name="close_jar")

# test where P3 _is_in_line of P1 and P2
P1 = np.array([1, 0, 0])
P2 = np.array([-1, 0, 0])
P3 = np.array([0, 0, 0])
assert augmentor._is_in_line(P3, P1, P2)




# test perturb_R_in_axis
R = np.quaternion(1, 0, 0, 0)
print(quaternion.as_rotation_matrix(R))
axis = np.array([0, 0, 1])
new_R = augmentor.perturb_R_around_axis(R, axis, **augmentor.cfg.fine_alignment.rotation_noise)
# print(new_R)
mat = quaternion.as_rotation_matrix(new_R)
# print(np.arctan2(mat[1, 0], mat[0, 0]) * 180 / np.pi)
zyz = quaternion.as_euler_angles(new_R) * 180 / np.pi
print(f"z {zyz[0]:.2f}, y {zyz[1]:.2f}, z {zyz[2]:.2f}")


# test perturb_T_in_plane
T_src =  np.array([0.473, 0.259, 0.92801 ])
T_tgt = np.array([0.473, 0.259, 0.75831 ])
T_noise = augmentor.perturb_T_in_plane(T_src, T_tgt-T_src, **augmentor.cfg.fine_alignment.translation_noise)
print(T_noise - T_src)



quan1 = np.quaternion(1, 0, 0, 0)
mat1 = quaternion.as_rotation_matrix(quan1)
print(mat1)
# rotate around z axis 90 degree
quan2 = np.quaternion(np.cos(np.pi/4), 0, 0, np.sin(np.pi/4))
mat2 = quaternion.as_rotation_matrix(quan2)
print(mat2)

print(quaternion.as_euler_angles(quan1)* 180 / np.pi)
print(quaternion.as_euler_angles(quan2)* 180 / np.pi)

q = quaternion.slerp_evaluate(quan1, quan2, 0.5)
print(q)
print(quaternion.as_euler_angles(q)* 180 / np.pi)