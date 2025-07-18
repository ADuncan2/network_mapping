from shapely.geometry import LineString, Point, shape, MultiPoint
from shapely.ops import split,nearest_points
import matplotlib.pyplot as plt


def interpolate_coordinates(coordinates, num_new_points):
    new_coordinates = []
    for i in range(len(coordinates) - 1):
        start_point = coordinates[i]
        end_point = coordinates[i + 1]

        direction_vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
        

        for step in range(0, num_new_points):
            interpolation_factor = step / (num_new_points-1)
            #print(interpolation_factor)
            new_x = start_point[0] + (interpolation_factor * direction_vector[0])
            new_y = start_point[1] + (interpolation_factor * direction_vector[1])
            if step == num_new_points:
                new_x = end_point[0]
                new_y = end_point[1]
            new_coordinates.append((new_x, new_y))

    return new_coordinates


def lines_segmentation(
        linestring: LineString,
        points: dict
        ) -> tuple[list, list]:
    sub_lines = []
    bookends = []
    coordinates = list(linestring.coords)
    coordinates_interp = interpolate_coordinates(coordinates,5)
    insert_indices = dict()
    count = 0
    for fid, geom in points.items():
        if geom.geom_type == "Polygon":
            coordinates_interp[0] = shape(geom).centroid
            insert_indices[0] = fid
            
        else:
            print(fid)
            min_distance = float('inf')
            closest_index = None
            for i, coord in enumerate(coordinates_interp):
                distance = Point(coord).distance(geom)
                if distance < min_distance:
                    min_distance = distance
                    closest_index = i
            #print(closest_index)
            coordinates_interp[closest_index] = geom.coords[0]
            insert_indices[closest_index] = fid
            count = count + 1
    sorted_dict = dict(sorted(insert_indices.items(), key=lambda item: item[0]))



    print(sorted_dict)
    return sub_lines, bookends


def subdivide_linestring(line, points):
    # Convert the points to a MultiPoint geometry
    points_geom = list(points.values())

    # Find the closest point on the line for each given point
    closest_points = [nearest_points(line, point)[0] for point in points_geom]
    for point in closest_points:
        print(point.distance(line))

    multi_point = MultiPoint(closest_points)
    lines = split(line,multi_point)

    return lines


def add_points_and_divide_linestring(
        linestring: LineString,
        points: dict
        ) -> tuple[list, list]:
    """
    Function to take an edge that has multiple
    junctions on it, given in the points dict, 
    and subdivide it into smaller regular edges.
    """
    coordinates = list(linestring.coords)
    insert_indices = dict()
    bookends = []

    #print(f"len ratio (coord/point): {len(coordinates)/len(points)}")
    if len(coordinates) <= len(points):
        print(f"len ratio (coord/point): {len(coordinates)/len(points)}")
        coordinates = interpolate_coordinates(coordinates,10)
        print(f"NEW len ratio (coord/point): {len(coordinates)/len(points)}")
    
## breaks here for some reason if more lines are interpolated. No error, no print out message, just ends the script.
## why would having more coordinates in a line, with the exact same start and end points, cause this for some but not all lines??

    count = 0
    for fid, geom in points.items():
        if geom.geom_type == "Polygon":
            coordinates[0] = shape(geom).centroid
            insert_indices[0] = fid
            
        else:
            min_distance = float('inf')
            closest_index = None
            for i, coord in enumerate(coordinates):
                distance = Point(coord).distance(geom)
                if distance < min_distance:
                    min_distance = distance
                    closest_index = i
            print(closest_index)
            coordinates[closest_index] = geom.coords[0]
            insert_indices[closest_index] = fid
            count = count + 1
    # Divide the LineString into smaller LineStrings
    #print(f"count: {count}")
    linestrings = []
    ordered_points = []
    start_index = 0

    #if this isn't ordered then why would we start with insert_indices[0]??
    ordered_points.append(insert_indices[0])

    #this should run through the points on the line, starting from one end (which end?) and then adds each substring in order
    #it should output a list of all the points apart from the end one at index i=0, does it?
    if len(points) != len(insert_indices):
        print(f"something missing! points: {len(points)}, insert_ind: {len(insert_indices)}")

    for i in range(len(coordinates)-1):
        if i > 0 and i in insert_indices.keys():
            linestrings.append(LineString(coordinates[start_index:i+1]))
            #print(f"i={i}")
            #print(f"insert_indices:{insert_indices[i]}")
            ordered_points.append(insert_indices[i])
            start_index = i
            #print("point added!")
            #print(insert_indices[i])
            #print(f"i:{i}")
    linestrings.append(LineString(coordinates[start_index:]))
    
    #print(ordered_points)
    
    outstanding = [z for z in list(points.keys()) if z not in ordered_points]

    ordered_points.append(outstanding[0])
    #print(ordered_points)
    
    bookends = []
    for i in range(len(ordered_points) - 1):
        bookends.append((ordered_points[i], ordered_points[i+1]))
    #print("subdivided linestrings:")
    #print(linestrings)
    #print("bookends of substring:")
    #print(bookends)
    return linestrings, bookends



# Create a LineString
coordinates_line = [
    (380817.8652842624, 412254.6490907022),
    (380811.96428859315, 412254.4200817352),
    (380807.827291678, 412254.22807551076),
    (380799.43629731255, 412254.22506204876),
    (380777.6333144332, 412252.6400302783),
    (380767.267322304, 412252.04201477324)
]
line = LineString(coordinates_line)
points = {
    14566409: Point(380817.869, 412254.649),
    14576895: Point(380808.999, 412254.282),
    14566405: Point(380811.964, 412254.42),
    14566400: Point(380815.662, 412254.563),
    14574986: Point(380767.267, 412252.042)
}

bookends = lines_segmentation(line,points)


# #Subdivide the LineString
# subdivided_lines = subdivide_linestring(line, points)

# #Print the resulting LineStrings
# print(subdivided_lines)



# # Plot the LineString
# plt.plot(*line.xy, label='LineString', marker='o')

# # Plot the points
# for fid, point in points.items():
#     plt.scatter(*point.xy, label=f'Point {fid}', marker='x')

# # Add labels and legend
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('LineString and Points Plot')
# plt.legend()

# # Show the plot
# plt.show()





