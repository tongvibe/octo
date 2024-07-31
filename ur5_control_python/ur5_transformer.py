# import numpy as np

# # 从标记到机器人基座的变换矩阵 (base_T_hand)
# T_tag2base = np.array([
#     [-0.6889793,   0.7238105,   0.03749518, -0.39535773],
#     [ 0.72211162,  0.68108662,  0.12114381, -0.4164536 ],
#     [ 0.0621477,   0.11054128, -0.99192655,  0.0384062 ],
#     [ 0.       ,   0.        ,  0.       ,   1.        ],
# ])

# # 从标记到相机的变换矩阵 (cam_T_tag)
# T_tag2cam = np.array([
#     [ 0.06530149,  0.99782885, -0.00856139,  0.04152845],
#     [-0.854388,    0.06034253,  0.51612007, -0.01945063],
#     [ 0.51551611, -0.02638866,  0.85647345,  0.86405996],
#     [ 0.       ,   0.        ,  0.      ,    1.        ]
# ])

# # 计算 T_tag2cam (cam_T_tag 的逆矩阵)
# T_base2tag = np.linalg.inv(T_tag2base)

# # 计算 T_base2cam
# T_base2cam = np.dot(T_base2tag, T_tag2cam)

# # 打印结果
# print("T_base2cam (robot base to camera):")
# print(T_base2cam)

# T_tag2cam_1 = np.array([
#     [ 0.08557555,  0.99626767, -0.01129364, -0.2131522 ],
#     [-0.85943683,  0.07954714,  0.50501543, -0.03522292],
#     [ 0.50402892, -0.0335108,   0.86303643,  0.87632469],
#     [ 0.      ,    0.       ,   0.        ,  1.        ]
# ])
# T_cam2base = np.linalg.inv(T_base2cam)
# T_tag2base_1 = np.dot(T_tag2cam_1, T_cam2base)
# print("new tag:")
# print(T_tag2base_1)


# # T_base2cam (robot base to camera):
# # [[-0.83115602  0.00323748  0.55602983  0.34102513]
# #  [ 0.03300415  0.99850721  0.04352095  0.52982237]
# #  [-0.5550589   0.05452399 -0.83002214 -0.50588425]
# #  [ 0.          0.          0.          1.        ]]

import numpy as np

# def compute_tag_to_base(T_base2cam, T_new_tag_to_cam):
#     # 计算从相机到新标记的逆变换矩阵
#     T_new_cam_to_tag = np.linalg.inv(T_new_tag_to_cam)
    
#     # 计算从机器人基座到新标记的变换矩阵
#     T_base_to_new_tag = np.dot(T_base2cam, T_new_cam_to_tag)
    
#     return T_base_to_new_tag

# 示例新的tag到相机的变换矩阵 (T_new_tag_to_cam)
T_new_tag_to_cam = np.array([
    [ 0.9814967,  -0.16701085 ,-0.09365682,  0.03951722],
    [ 0.05400776,  0.71072483, -0.70139388 ,-0.03574346],
    [ 0.18370462,  0.68335759 , 0.70659396,  0.42152664],
    [ 0.         , 0.    ,      0.      ,    1.        ]
])

T_base2cam = np.array([
    [-0.06687333,  0.76151493, -0.25466243,  0.2011803 ],
    [-0.1472856,  -0.33070181, -0.7493762 , -0.12398504],
    [-0.91515842, -0.01838072 , 0.16388952 ,-0.40158824],
    [ 0.      ,    0.      ,    0.       ,   1.        ]
])

T_cam2base = np.linalg.inv(T_base2cam)
T_new_tag2base = np.dot(T_new_tag_to_cam, T_cam2base)
# 计算新的标签到机器人基座的变换矩阵
# T_base_to_new_tag = compute_tag_to_base(T_base2cam, T_new_tag_to_cam)
# [-0.07637649 -0.48457747 -0.87140766 -0.30239903]
#  [ 0.81463743  0.473602   -0.3347641  -0.36787584]
#  [ 0.57491955 -0.7354494   0.35858289  0.22975185]
#  [ 0.          0.          0.          1.        ]
# 打印结果
print("T_base_to_new_tag:")
print(T_new_tag2base)