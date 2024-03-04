import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import pyvista as pv
from sympy import symbols, Eq, solve
from sympy.geometry import Circle, Point3D, Plane
import math


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


def get_geometric_objects_cell(geometric_objects):
    for i in range(geometric_objects.n_cells):
        yield geometric_objects.get_cell(i)


def find_normal_vector(point1, point2, point3):
    vec1 = np.array(point2) - np.array(point1)
    vec2 = np.array(point3) - np.array(point1)
    cross_vec = np.cross(vec1, vec2)
    return cross_vec / np.linalg.norm(cross_vec)


def get_viewed_area():
    # Create some sample meshes
    pos_mesh = np.array([1, 0, 0])
    r_mesh = 1

    mesh = pv.Sphere(radius=r_mesh, center=pos_mesh)
    mesh1 = pv.Box(bounds=(-5.0, -4.0, -1.0, 1.0, -1.0, 1.0))

    # Create a plotter
    plotter = pv.Plotter()

    cy_direction = np.array([0, 0, 1])
    cy_hight = 0.875
    n_resolution = 36

    # Calculate the length of the lateral surface of an inscribed cylinder
    h = np.cos(np.pi / n_resolution) * r_mesh
    l = np.sqrt(np.abs(4 * h ** 2 - 4 * r_mesh ** 2))

    # Find the radius of the spheres
    z_resolution = int(np.ceil(cy_hight / l))
    h = cy_hight / z_resolution
    spheres_radius = np.max([l, h]) / 2

    cylinder = pv.CylinderStructured(center=pos_mesh, direction=cy_direction, radius=r_mesh, height=cy_hight,
                                     theta_resolution=n_resolution, z_resolution=z_resolution)

    # Create the hemispheres and add them to the faces of the cylinder
    for cell in get_geometric_objects_cell(cylinder):
        pos_cell = cell.center
        points_cell = cell.points[:3]
        norm_vec = find_normal_vector(*points_cell)

        sub_mesh = pv.Sphere(radius=spheres_radius, center=pos_cell, direction=norm_vec, end_phi=90)
        plotter.add_mesh(sub_mesh)

    # cylinder.plot(show_edges=True)
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

    direction = np.array(plotter.camera.focal_point) - np.array(plotter.camera.position)
    direction /= np.linalg.norm(direction)

    A, B, C = direction

    direction = -direction

    dot_plane = np.array([0.5, 0, 0])

    D = -np.dot(direction, dot_plane)

    print(f'{A}x + {B}y + {C}z + {D} = 0')
    plane = pv.Plane(dot_plane, direction, i_size=5, j_size=5)
    # mesh.plot(show_edges=True)
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
                points1[c * 2, c_even] = bounds_line[j]
                c_even += 1
            else:
                points1[(c * 2) + 1, c_odd] = bounds_line[j]
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


# Define variables
x, y, z, t = symbols('x y z t')


def get_line_of_intersection_two_planes(pi1, pi2):
    # Define the equations of the planes
    plane1 = Eq(pi1[0] * x + pi1[1] * y + pi1[2] * z, -pi1[3])
    plane2 = Eq(pi2[0] * x + pi2[1] * y + pi2[2] * z, -pi2[3])

    # Solve the system of equations to find the direction vector
    direction_vector = np.cross(pi1[:3], pi2[:3])

    # Find a point on the line of intersection (by setting one variable to zero)
    # Here we set z = 0, you can choose any other variable as well
    point = solve((plane1, plane2))
    point[x] = point[x].subs(z, 0)
    point[y] = point[y].subs(z, 0)

    # Formulate the parametric equation of the line
    parametric_equation = [point[x] + direction_vector[0] * t,
                           point[y] + direction_vector[1] * t,
                           direction_vector[2] * t]
    return parametric_equation


def get_intersection_points_line_sphere(line_parametric_eq, sphere_eq):
    # Extract components of the line's parametric equations
    x_expr, y_expr, z_expr = line_parametric_eq

    # Extract components of the sphere equation
    x_sphere, y_sphere, z_sphere, r = sphere_eq

    # Substitute the parametric equations of the line into the equation of the sphere
    sphere_eq_subs = Eq((x_expr - x_sphere) ** 2 + (y_expr - y_sphere) ** 2 + (z_expr - z_sphere) ** 2, r ** 2)

    # Solve for t to find the point(s) of intersection
    solutions = solve(sphere_eq_subs, t)

    # Evaluate the parametric equations at the intersection point(s)
    intersection_points = np.empty([0, 3])
    for sol in solutions:
        x_inter = x_expr.subs(t, sol)
        y_inter = y_expr.subs(t, sol)
        z_inter = z_expr.subs(t, sol)
        intersection_points = np.row_stack((intersection_points, (float(x_inter), float(y_inter), float(z_inter))))

    return intersection_points


def spherical_distance(p1, p2):
    """
    Calculate the spherical distance between two points on a unit sphere.
    """
    # Convert spherical coordinates to radians
    lon1, lat1 = math.radians(p1[0]), math.radians(p1[1])
    lon2, lat2 = math.radians(p2[0]), math.radians(p2[1])

    # Calculate spherical distance using haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = c

    return distance


def spherical_triangle_area(p1, p2, p3):
    """
    Calculate the area of a spherical triangle formed by three points on a unit sphere.
    """
    # Calculate the lengths of the three sides of the spherical triangle
    side1 = spherical_distance(p1, p2)
    side2 = spherical_distance(p2, p3)
    side3 = spherical_distance(p3, p1)

    # Calculate the semi-perimeter
    s = (side1 + side2 + side3) / 2

    # Calculate the spherical excess using Heron's formula
    excess = 4 * math.atan(
        math.sqrt(math.tan(s / 2) * math.tan((s - side1) / 2) * math.tan((s - side2) / 2) * math.tan((s - side3) / 2)))

    # The area of the spherical triangle is equal to its excess angle
    area = excess

    return area


def plane_circle_intersection(plane_eq, circle):
    # Extract components of the plane equation
    a, b, c, d = plane_eq

    # Define the plane
    plane = Plane(Point3D(0, 0, -d/c), normal_vector=(a, b, c))

    # Project the circle onto the plane
    projected_circle = circle.projection(plane)

    # Find the intersection points between the projected circle and the plane
    intersection_points_ci = projected_circle.intersection(plane)

    return intersection_points_ci


def get_viewed_area_from():
    print('Starting viewed area computing')


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
    # get_viewed_area()  # Only function used with pyvista

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
    pi1gl = np.array([2.0, 4.0, -1.0, 1.0])
    pi2gl = np.array([-1.0, 2.0, 1.0, 2.0])
    parametric_equation = get_line_of_intersection_two_planes(pi1gl, pi2gl)
    print(parametric_equation)
    xl = parametric_equation[0].subs(t, 0.0)
    yl = parametric_equation[1].subs(t, 0.0)
    zl = 0.0
    sphere_eq = (xl, yl, zl, 2)
    # Find intersection point(s)
    intersection_points = get_intersection_points_line_sphere(parametric_equation, sphere_eq)

    # Display intersection point(s)
    print("Intersection point(s) with the sphere:")
    for point in intersection_points:
        print(point)

    distance = np.linalg.norm(intersection_points[0] - intersection_points[1])

    print(distance)
