import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import pyvista as pv


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def plot_circle(radius: float, resolution: float, ax=None):
    # Generate points along the circumference of the circle
    theta = np.linspace(0, 2 * np.pi, resolution)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(x)

    # Plot the circle in 3D
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color='blue', linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Circle in 3D')
    return ax


def plot_circle_in_plane(normal, center, radius, ax=None):
    # Unpack the normal vector and center point
    a, b, c = normal
    x0, y0, z0 = center

    # Generate a unit vector orthogonal to the normal vector
    u = np.array([1, 0, 0])
    if np.dot(u, normal) == 1:
        u = np.array([0, 1, 0])
    u = u - np.dot(u, normal.T) * normal
    u = u / np.linalg.norm(u)

    # Generate another unit vector orthogonal to the normal vector and u
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    # Generate points along the circumference of the circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_points = np.array([
        x0 + radius * (np.cos(theta) * u[0] + np.sin(theta) * v[0]),
        y0 + radius * (np.cos(theta) * u[1] + np.sin(theta) * v[1]),
        z0 + radius * (np.cos(theta) * u[2] + np.sin(theta) * v[2])
    ])

    if ax is None:
        # Plot the circle in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(circle_points[0], circle_points[1],
                circle_points[2], color='blue', linewidth=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Circle in Plane with Normal Vector')
    else:
        ax.plot(circle_points[0], circle_points[1],
                circle_points[2], color='blue', linewidth=2)
    return ax


def plot_line(vector, point, ax):
    # Unpack the vector and point
    a, b, c = vector
    x0, y0, z0 = point

    # Define the range of t
    t = np.linspace(-1, 1, 100)

    # Calculate points on the line
    x = x0 + a * t
    y = y0 + b * t
    z = z0 + c * t

    if ax is None:
        # Plot the line in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x0, y0, z0, color='red', marker='-o',
                   label='Point (x0, y0, z0)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Line in 3D')
        ax.legend()
    else:
        ax.plot(x, y, z, color='blue', linewidth=2)
    return ax


def plot_plane_through_points(point1, point2, point3, ax=None):
    # Define the points
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    x3, y3, z3 = point3

    # Calculate the normal vector
    v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
    v2 = np.array([x3 - x1, y3 - y1, z3 - z1])
    normal = np.cross(v1, v2)

    # Define a grid of points to plot the plane
    xx, yy = np.meshgrid(np.linspace(x1 - 1, x3 + 1, 10),
                         np.linspace(y1 - 1, y3 + 1, 10))
    zz = (-normal[0] * (xx - x1) - normal[1] * (yy - y1)) / normal[2] + z1

    if ax is None:
        # Plot the plane in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, zz, alpha=0.5)

        # Plot the points
        ax.scatter(x1, y1, z1, color='red', marker='o', label='Point 1')
        ax.scatter(x2, y2, z2, color='green', marker='o', label='Point 2')
        ax.scatter(x3, y3, z3, color='blue', marker='o', label='Point 3')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Plane through Three Points')
        ax.legend()
    else:
        ax.scatter(x1, y1, z1, color='red', marker='o', label='Point 1')
        ax.scatter(x2, y2, z2, color='green', marker='o', label='Point 2')
        ax.scatter(x3, y3, z3, color='blue', marker='o', label='Point 3')
    return normal, ax


def plot_hemisphere(center, radius, theta, ax=None):
    # Unpack the center
    a, b, c = center

    # Generate points on the hemisphere surface
    phi = np.linspace(0, np.pi, 50)
    theta_rad = np.deg2rad(theta)
    # x = a + radius * np.outer(np.sin(phi), np.cos(theta_rad))
    # y = b + radius * np.outer(np.sin(phi), np.sin(theta_rad))
    # z = c + radius * np.outer(np.cos(phi), np.ones_like(theta_rad))

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi / 2, 50)
    x = a + radius * np.outer(np.cos(u), np.sin(v))
    y = b + radius * np.outer(np.sin(u), np.sin(v))
    z = c + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    if ax is None:
        # Plot the hemisphere in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, color='blue', alpha=0.8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Hemisphere')
    else:
        ax.plot_surface(x, y, z, color='blue', alpha=0.8)
    return ax, x, y, z


def compute_orientation(normal):
    # Normalize plane normal
    normal = normal / np.linalg.norm(normal)

    # Compute the quaternion orientation
    # From the plane normal
    rot = Rotation.from_rotvec(np.pi / 2 * normal)
    return rot


def get_viewed_area():
    # Create some sample meshes
    pos_mesh = np.array([1, 0, 0])
    r_mesh = 1

    mesh = pv.Sphere(radius=r_mesh, center=pos_mesh)
    mesh1 = pv.Box(bounds=(-5.0, -4.0, -1.0, 1.0, -1.0, 1.0))

    # Create a plotter
    plotter = pv.Plotter()

    cy_direction = np.array([0, 0, 1])
    cy_hight = 5
    n_resolution = 36
    h = np.sin(180/n_resolution) * r_mesh
    l = np.sqrt(np.abs(4*h**2 - 4*r_mesh**2))

    z_resolution = int(np.ceil(cy_hight / l))

    cylinder = pv.CylinderStructured(center=pos_mesh, direction=cy_direction, radius=r_mesh, height=cy_hight, theta_resolution=n_resolution, z_resolution=z_resolution)
    cylinder.plot(show_edges=True)
    # Add the meshes to the plotter
    # plotter.add_mesh(mesh1)
    # plotter.add_mesh(mesh)
    plotter.add_mesh(cylinder)

    # Set camera position and orientation (optional)
    plotter.camera.clipping_range = (1e-4, 1)
    # plotter.camera_position = [(10, 0, 0), (0, 0, 0), (0, 0, 0)]

    points = np.array([[2.0, 0.0, 0.0], [2.0, 2.0, 0.0],
                      [2.0, 0.0, 2.0], [2.0, 2.0, 2.0]])
    point_cloud = pv.PolyData(points)
    plotter.add_mesh(point_cloud)

    # Get the camera's view frustum
    frustum = plotter.camera.view_frustum()
    plotter.add_mesh(frustum, style="wireframe")

    direction = np.array(plotter.camera.focal_point) - \
        np.array(plotter.camera.position)
    direction /= np.linalg.norm(direction)

    focal_point = np.array(plotter.camera.focal_point)

    A, B, C = direction

    direction = -direction

    dot_plane = np.array([0.5, 0, 0])

    D = -np.dot(direction, dot_plane)

    print(f"{A}x + {B}y + {C}z + {D} = 0")
    plane = pv.Plane(dot_plane, direction, i_size=5, j_size=5)
    mesh.plot(show_edges=True)
    plotter.add_mesh(plane, color="red", opacity=0.2)

    p = np.dot(direction, pos_mesh - dot_plane) / np.linalg.norm(direction)
    print(f'{p=}')

    A = 2 * np.pi * r_mesh * (r_mesh - p)
    print(f'{A=}')

    bounds_mesh = mesh.bounds
    # Get the bounds of the meshes
    # bounds_mesh1 = mesh1.bounds

    # Calculate the intersection of the camera frustum and mesh bounds to find the viewed area
    # viewed_area_mesh = [max(bounds_mesh[0], frustum.bounds[0]), min(bounds_mesh[1], frustum.bounds[1]),
    #                      max(bounds_mesh[2], frustum.bounds[2]), min(bounds_mesh[3], frustum.bounds[3]),
    #                      max(bounds_mesh[4], frustum.bounds[4]), min(bounds_mesh[5], frustum.bounds[5])]
    bellow_plane = frustum.get_cell(0)
    above_plane = frustum.get_cell(1)
    right_plane = frustum.get_cell(2)
    left_plane = frustum.get_cell(3)  # Get a plane of the frustum
    far_clip = frustum.get_cell(4)
    near_clip = frustum.get_cell(5)
    points1 = np.empty((4, 3))
    c = 0
    for i in range(3):
        if i == 1:
            continue
        # Get each line on the border of the plane of the frustum
        line = above_plane.get_edge(i)
        # line.bounds get 6 numbers of the line [x_start,x_end,y_start,y_end,z_start,z_end]
        bounds_line = np.array(line.bounds)
        c_odd = 0
        c_even = 0
        for j in range(6):
            if j % 2:
                points1[c*2, c_even] = bounds_line[j]
                c_even += 1
            else:
                points1[(c*2)+1, c_odd] = bounds_line[j]
                c_odd += 1
        c += 1

    # mesh = pv.Plane(center=point1, direction=normal, i_size=15, j_size=15)
    # Create a plane from the points (not working yet) Works with variable points but does not work with points1
    mesh_plane = create_mesh_from_points(points1)
    plotter.add_mesh(mesh_plane)  # Add the mesh of the plane to plotter figure
    plotter.show_grid()
    # Show the plotter
    plotter.show()

    if far_clip.bounds[0] < mesh.bounds[1] < near_clip.bounds[0]:
        print('Sphere viewed')

    viewed_area_mesh = np.mean(np.array([(bounds_mesh[0] - frustum.bounds[0]), (bounds_mesh[1] - frustum.bounds[1]),
                                         (bounds_mesh[2] - frustum.bounds[2]
                                          ), (bounds_mesh[3] - frustum.bounds[3]),
                                         (bounds_mesh[4] - frustum.bounds[4]), (bounds_mesh[5] - frustum.bounds[5])]))

    # print("Viewed area of mesh1:", viewed_area_mesh1)
    print("Viewed area of mesh2:", viewed_area_mesh)


def create_mesh_from_points(points):
    # Create a mesh from four points
    mesh = pv.PolyData()
    mesh.points = np.array(points[0])  # point[]
    mesh.faces = np.array([4, 0, 1, 2, 3], np.int8)
    return mesh


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ax = plot_circle(1.0, 500)
    # vector = np.array((0, 1, 1))
    # point = np.array((0, 0, 0))
    # radius = 1.0
    # ax = plot_circle_in_plane(vector, point, radius)
    # Example usage
    # point1 = np.array((2, 2, 3))
    # point2 = np.array((4, 5, 6))
    # point3 = np.array((7, 7, 9))
    # get_viewed_area()
    # normal, ax = plot_plane_through_points(point1, point2, point3)
    # ax = plot_circle_in_plane(normal, point1, radius, ax)
    # ax = plot_line(normal, point1, ax)
    # ax, x, y, z = plot_hemisphere(point1, radius, 0.0, ax)
    # rot = compute_orientation(normal)
    # for i in range(x.shape[0]):
    #     Points = np.column_stack((x[:, i], y[:, i], z[:, i]))
    #     rot_points = rot.apply(Points)
    #     ax.scatter(rot_points[:, 0], rot_points[:, 1], rot_points[:, 2])
    # plt.show()
    get_viewed_area()  # Only function used with pyvista
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
