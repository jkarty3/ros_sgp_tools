import utm
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Polygon, LineString, Point

try:
    from std_msgs.msg import Header
    from sensor_msgs.msg import PointCloud2, PointField
except ImportError:
    pass

# Extract geofence and home location from QGC plan file
def get_mission_plan(fname, get_waypoints=False):
    with open(fname, "r") as infile:
        data = json.load(infile)
        vertices = np.array(data['geoFence']['polygons'][0]['polygon'])
        home_position = data['mission']['plannedHomePosition']
        if get_waypoints:
            waypoints = []
            for waypoint in data['mission']['items'][1]['TransectStyleComplexItem']['Items']:
                if waypoint['command']==16:
                    waypoints.append(waypoint['params'][4:7])
            return vertices, home_position, np.array(waypoints)
    return vertices, home_position

def point_cloud(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx3 array of xyz positions.
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message

    Code source:
        https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
    """
    # In a PointCloud2 message, the point cloud is stored as an byte 
    # array. In order to unpack it, we also include some parameters 
    # which desribes the size of each individual point.
    points = np.array(points, dtype=np.float32)
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.

    data = points.astype(dtype).tobytes() 

    # The fields specify what the bytes represents. The first 4 bytes 
    # represents the x-coordinate, the next 4 the y-coordinate, etc.
    fields = [PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyz')]

    # The PointCloud2 message also has a header which specifies which 
    # coordinate frame it is represented in. 
    header = Header(frame_id=parent_frame)

    return PointCloud2(
        header=header,
        height=1, 
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 3), # Every point consists of three float32s.
        row_step=(itemsize * 3 * points.shape[0]),
        data=data
    )

class LatLonStandardScaler(StandardScaler):
    def fit(self, X, y=None, sample_weight=None):
        # Map lat long to UTM points before normalization
        X = utm.from_latlon(X[:, 0], X[:, 1])
        self.encoding = X[2:]
        X = np.vstack([X[0], X[1]]).T

        # Fit normalization params
        super().fit(X, y=y, sample_weight=sample_weight)

        # Change variance/scale parameter to ensure all axis are scaled to the same value
        ind = np.argmax(self.var_)
        self.var_ = np.ones(X.shape[-1])*self.var_[ind]
        self.scale_ = np.ones(X.shape[-1])*self.scale_[ind]

    def transform(self, X, copy=None):
        # Map lat long to UTM points before normalization
        X = utm.from_latlon(X[:, 0], X[:, 1])
        X = np.vstack([X[0], X[1]]).T
        return super().transform(X, copy=copy)
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def inverse_transform(self, X, copy=None):
        X = super().inverse_transform(X, copy=copy)

        # Map UTM to lat long points after de-normalization
        X = utm.to_latlon(X[:, 0], X[:, 1], 
                          self.encoding[0], self.encoding[1])
        X = np.vstack([X[0], X[1]]).T
        return X


# Calculate the intersections between the line (point a -> point a) and the polygon, then order them by distance to point a
def ordered_intersections_by_distance(point_a, point_b, polygon):
    line = LineString([point_a, point_b])
    intersection = polygon.intersection(line)

    if isinstance(intersection, LineString):
        intersection_points = list(intersection.coords)
    elif hasattr(intersection, 'geoms'):
        intersection_points = [pt for geom in intersection.geoms for pt in geom.coords]
    else:
        intersection_points = []

    return sorted(intersection_points, key=lambda p: Point(p).distance(Point(point_a)))

# Finds the indices to add an external line to a polygon to create
def find_insertion_index(exterior_coords, intersection_point):
    min_distance = float('inf')
    insertion_index = -1

    for i in range(len(exterior_coords) - 1):
        segment_start = Point(exterior_coords[i])
        segment_end = Point(exterior_coords[i + 1])

        segment = LineString([segment_start, segment_end])
        distance = segment.distance(Point(intersection_point))

        if distance < min_distance:
            min_distance = distance
            insertion_index = i 

    return insertion_index

# Get 2 points inside the polygon that are near where the line (point a -> point b) intersects the polygon
def get_padded_intersects(point_a, point_b, fence_polygon, padding_length=0.0001):
    line = LineString([point_a, point_b])
    intersection_coords = ordered_intersections_by_distance(point_a, point_b, fence_polygon)
    intersection_coords = [intersection_coords[1], intersection_coords[-2]]

    padded_points = []
    for i in range(len(intersection_coords)):
        intersection_point = Point(intersection_coords[i])
        dist_to_a = line.project(intersection_point)
        if dist_to_a<padding_length or dist_to_a + padding_length > line.length:
            continue
        if i==0:
            padded_point = line.interpolate(dist_to_a - padding_length)
        else:
            padded_point = line.interpolate(dist_to_a + padding_length)
        
        padded_points.append((padded_point.x, padded_point.y))

    return padded_points

# Move one point from the outside of the polygon back to the inside
def get_moved_point(point_a, point_b, fence_polygon, padding_length=0.0001):
    intersection_coords = ordered_intersections_by_distance(point_a, point_b, fence_polygon)[1:-1]

    # Calculate the two polygons that this intersecting line creates with the origional polygon
    exterior_coords = list(fence_polygon.exterior.coords)
    insertion_index_1 = find_insertion_index(exterior_coords, intersection_coords[0])
    insertion_index_2 = find_insertion_index(exterior_coords, intersection_coords[1])
    polygon_1 = Polygon(exterior_coords[:insertion_index_1+1] + intersection_coords[:2] + exterior_coords[insertion_index_2:])
    polygon_2 = Polygon(exterior_coords[insertion_index_2:insertion_index_1:-1] + intersection_coords[:2] + [exterior_coords[insertion_index_2]])
    larger_polygon = polygon_1 if polygon_1.area > polygon_2.area else polygon_2

    # Find the correct direction to move the midpoint (towards the larger polygon)
    dx, dy = intersection_coords[1][0]-intersection_coords[0][0], intersection_coords[1][1]-intersection_coords[0][1]
    normal_vector = np.array([dy, -dx]) / np.linalg.norm(np.array([-dy, dx]))
    midpoint = Point((intersection_coords[0][0]+intersection_coords[1][0])/2, (intersection_coords[0][1]+intersection_coords[1][1])/2)
    moved_midpoint = Point(midpoint.x + normal_vector[0] * 0.00000001, midpoint.y + normal_vector[1] * 0.00000001)
    if not larger_polygon.contains(moved_midpoint):
        normal_vector = -normal_vector

    # Then move it to the intersect of the smaller polygon and add padding
    far_point = Point(midpoint.x + normal_vector[0], midpoint.y + normal_vector[1])
    intersections = ordered_intersections_by_distance(Point(midpoint), far_point, fence_polygon)
    intersect = Point(intersections[0][0], intersections[0][1])
    padded = Point(intersect.x + normal_vector[0] * padding_length, intersect.y + normal_vector[1] * padding_length)
    while not (fence_polygon.contains(padded)) and len(ordered_intersections_by_distance(intersect, padded, fence_polygon))<4:

        padding_length /= 2
        padded = Point(intersect.x + normal_vector[0] * padding_length, intersect.y + normal_vector[1] * padding_length)
    return (padded.x, padded.y)

# Calculate a list of points to travel from point a to point b while staying inside the polygon
def calculate_bounded_path(point_a, point_b, fence_polygon, padding_length=0.0009):
    path = [point_a, point_b]

    if not fence_polygon.contains(LineString([path[0], path[1]])):
        # First calculate padded points close to the intersection
        padded_intersects = get_padded_intersects(path[0], path[1], fence_polygon, padding_length)
        temp = [path[0]]
        if len(padded_intersects):
            for intersect in padded_intersects:
                temp.append(intersect)
        temp.append(path[1])
        path = temp

        # Then move points that are outside the polygon back into the polygon
        i=0
        while i<len(path)-1:
            if fence_polygon.contains(LineString([path[i], path[i+1]])):
                i+=1
            else:
                moved_point = get_moved_point(path[i], path[i+1], fence_polygon, padding_length)
                path.insert(i+1, moved_point)

    return path
