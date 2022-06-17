from matplotlib import pyplot as plt

def rgba(r):
	"""Generates a color based on range.

	Args:
		r: the range value of a given point.
	Returns:
		The color for a given range
	"""
	c = plt.get_cmap('jet')((r % 50.0) / 50.0)
	c = list(c)
	c[-1] = 0.5  # alpha
	return c

def plot_image(camera_image):
	"""Plot a cmaera image."""
	plt.figure(figsize=(20, 12))
	plt.imshow(camera_image)
	plt.grid(visible=False)

def plot_points_on_image(projected_points, camera_image, title, rgba_func =rgba, point_size=5.0):
    """Plots points on a camera image.

    Args:
        projected_points: [N, 3] numpy array. The inner dims are
            [camera_x, camera_y, range].
        camera_image: jpeg encoded camera image.
        rgba_func: a function that generates a color from a range value.
        point_size: the point size.

    """
    plot_image(camera_image)

    xs = []
    ys = []
    colors = []

    for point in projected_points:
        xs.append(point[0])  # width, col
        ys.append(point[1])  # height, row
        colors.append(rgba_func(point[2]))

    plt.title(title)
    plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")
    plt.axis('off')
    plt.grid(visible=False)