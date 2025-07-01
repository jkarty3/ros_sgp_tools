import utm
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Polygon, LineString, Point
import heapq
import networkx as nx

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
        if dist_to_a<padding_length:
            padded_point = Point(point_a)
        elif dist_to_a + padding_length > line.length:
            padded_point = Point(point_b)
        elif i==0:
            padded_point = line.interpolate(dist_to_a - padding_length)
        else:
            padded_point = line.interpolate(dist_to_a + padding_length)
        
        padded_points.append((padded_point.x, padded_point.y))

    return padded_points

def lazy_astar(nodes, start_idx, goal_idx, polygon):
    open_set = []
    heapq.heappush(open_set, (0, start_idx))
    
    came_from = {}
    g_score = {start_idx: 0}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal_idx:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        for neighbor in range(len(nodes)):
            if neighbor == current:
                continue
            edge = LineString([nodes[current], nodes[neighbor]])
            if not (polygon.contains(edge) or polygon.touches(edge)):
                continue
            
            tentative_g = g_score[current] + np.linalg.norm(np.array(nodes[current]) - np.array(nodes[neighbor]))
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + np.linalg.norm(np.array(nodes[neighbor]) - np.array(nodes[goal_idx]))
                heapq.heappush(open_set, (f_score, neighbor))
    
    return None  

# Calculate a list of points to travel from point a to point b while staying inside the polygon
def calculate_bounded_path(point_a, point_b, fence_polygon, padding_length = 0.00005):
    

    # First calculate padded points close to the intersection
    padded_intersects = get_padded_intersects(point_a, point_b, fence_polygon, padding_length)

    # Then run lazy A* to find the best path
    nodes = list(fence_polygon.exterior.coords[:-1])
    nodes.append(padded_intersects[0])
    nodes.append(padded_intersects[1])
    G = nx.Graph()
    for i, p in enumerate(nodes):
        G.add_node(i, pos=p)

    start_idx = len(nodes) - 2
    goal_idx = len(nodes) - 1

    path = lazy_astar(nodes, start_idx, goal_idx, fence_polygon)
    if path is None:
        return None
    else:
        final_path = []
        for point in path:
            final_path.append(nodes[point])
        return final_path

