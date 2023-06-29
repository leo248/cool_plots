import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import math as m
points = {
    "A": (0, 0, 0),
    "B": (1, 0, 0),
    "C": (1, m.sqrt(3), 0),
    "D": (2, 0, 0),
    "E": (1.5, m.sqrt(3)/2, 0),
    "F": (0.5, m.sqrt(3)/2, 0),
    "G": (1, m.sqrt(3)/3, 0),
    "H": (1, m.sqrt(3)/3, m.sqrt(2+2/3)),
    "I": (1, m.sqrt(3)/3/3, m.sqrt(2+2/3)/3),
    "J": (0.5, m.sqrt(3)/3/2,  m.sqrt(2+2/3)/2),
    "K": (1.5, m.sqrt(3)/3/2,  m.sqrt(2+2/3)/2),
    "L": (8/12,m.sqrt(3)/9+m.sqrt(3)/3,  m.sqrt(2+2/3)/3),
    "M": (1, m.sqrt(3)*2/3, m.sqrt(2+2/3)/2),
    "P": (1+1/3, m.sqrt(3)/9+m.sqrt(3)/3, m.sqrt(2+2/3)/3)
    
}
point_in_the_middle = {"N": (1, m.sqrt(3)/3, ((points["L"][2]+points["P"][2]+points["B"][2])/3 + (points["M"][2]+points["I"][2]+points["G"][2])/3) / 2)}
points.update(point_in_the_middle)
df = pd.DataFrame(points, index=['X', 'Y', 'Z']).T

x = df['X']
y = df['Y']
z = df['Z']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z)
###################################################################
def plot_circle(ax, point_labels, circle_center):
    p1, p2, p3 = (np.array(points[label]) for label in point_labels)
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)

    a = np.linalg.norm(p3 - p2)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p2 - p1)
    s = (a + b + c) / 2.0
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    radius = a * b * c / (4.0 * area)

    u = (v1 - np.dot(v1, normal) * normal / np.linalg.norm(normal)**2)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    angles = np.linspace(0, 2 * np.pi, 100)
    circle_points = np.zeros((3, len(angles)))

    for i in range(len(angles)):
        angle = angles[i]
        x_offset = radius * np.cos(angle)
        y_offset = radius * np.sin(angle)
        circle_points[:, i] = circle_center + x_offset * u + y_offset * v

    ax.plot(circle_points[0, :], circle_points[1, :], circle_points[2, :])






plot_circle(ax, ['B', 'L', 'P'],(1,m.sqrt(3)/4,m.sqrt(3/2)/4))
plot_circle(ax, ["E", "I", "L"],(1.125,0.649519,0.306186))
plot_circle(ax,  ["F", "P", "I"],(0.875,0.649519,0.306186))
plot_circle(ax, ["G", "L", "K"],(1.125,0.505181,0.51031))
plot_circle(ax,  ["G", "I", "M"],(1,0.721688,0.51031))
plot_circle(ax, ["G", "P", "J"],(0.875,0.505181,0.51031))
###################################################################
# straight lines
lines = [
    ['A', 'B', 'D'],
    ['A', "F", 'C'],
    ['C', "E", 'D'],
    ['B', 'G', 'C'],
    ['A', 'G', 'E'],
    ["F", "G", "D"],
   ["A","J", "H"],
  ["D","K", "H"],
    ["C", "M", "H"],
    ["B","I","H"],
    ["J","I","D"],
    ["K","I","A"],
     ["F", "L","H"],
     ["C", "L","J"],
    ["A", "L", "M"],
    ["D", "P", "M"],
     ["C", "P", "K"],
     ["H", "P", "E"],
    ["A","N",  "P"],
     ["D","N",  "L"],
     ["H","N", "G"],
      ["J", "N", "E"],
   ["B", "N", "M"],
    ["K", "N", "F"],
    ["I", "N", "C"]
]


num_points = 100
theta = np.linspace(0, 2 * np.pi, num_points)

def plot_circle(plane, center_point, radius_point, color, ax):
    radius = np.linalg.norm(df.loc[center_point] - df.loc[radius_point])

    normal_vector = np.cross(df.loc[plane[1]] - df.loc[plane[0]], df.loc[plane[2]] - df.loc[plane[0]])

    basis1 = np.cross(normal_vector, df.loc[center_point] - df.loc[plane[0]])
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = np.cross(normal_vector, basis1)
    basis2 = basis2 / np.linalg.norm(basis2)

    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_points = df.loc[center_point].values[:, None] + radius * (np.outer(basis1, np.cos(theta)) + np.outer(basis2, np.sin(theta)))

    ax.plot(circle_points[0, :], circle_points[1, :], circle_points[2, :], color=color, linewidth=2)

center_between_I_G_M = (df.loc["I"] + df.loc["G"] + df.loc["M"])/3

input_dict = {
    ('A', 'H', 'D'): ('I', 'B', 'blue'),
    ('H', 'C', 'A'): ('L', 'F', 'green'),
    ('H', 'C', 'D'): ('P', 'E', 'purple'),
      ('A', 'D', 'C'): ('G', 'B', 'red'),

}
for plane, points in input_dict.items():
    plot_circle(plane, points[0], points[1], points[2], ax)



for line in lines:
    line_x = df.loc[line, 'X']
    line_y = df.loc[line, 'Y']
    line_z = df.loc[line, 'Z']
    ax.plot(line_x, line_y, line_z)
for point, coords in df.iterrows():
    ax.text(coords['X'], coords['Y'], coords['Z'], point)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
plt.axis('off')
plt.tight_layout()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0,0)

plt.show()
