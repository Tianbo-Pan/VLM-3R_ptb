OBJ_COUNT_TEMPLATE = """
How many {category}(s) are in this room?
""".strip()

OBJ_SIZE_ESTIMATE_TEMPLATE = """
What is the length of the longest dimension (length, width, or height) of the {category}, measured in centimeters?""".strip()

ROOM_SIZE_TEMPLATE = """
What is the size of this room (in square meters)? 
If multiple rooms are shown, estimate the size of the combined space.
""".strip()

OBJ_ABS_DISTANCE_TEMPLATE = """
Measuring from the closest point of each object, what is the direct distance between the {object1} and the {object2} (in meters)?
""".strip()

OBJ_REL_DISTANCE_V1_TEMPLATE = """
Measuring from the closest point of each object, which of these objects ({choice_a}, {choice_b}, {choice_c}, {choice_d}) is the closest to the {category}?
If there are multiple instances of an object category, measure to the closest.
""".strip()

OBJ_REL_DISTANCE_V2_TEMPLATE = """
Measuring from the closest point of each object, which of these objects ({choice_a}, {choice_b}, {choice_c}) is the closest to the {category}?
If there are multiple instances of an object category, measure to the closest.
""".strip()

OBJ_REL_DISTANCE_V3_TEMPLATE = """
Measuring from the closest point of each object, which of these objects ({choice_a}, {choice_b}) is the closest to the {category}?
If there are multiple instances of an object category, measure to the closest.
""".strip()

OBJ_REL_DIRECTION_V1_TEMPLATE = """
If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to my front-left, front-right, back-left, or back-right?
The directions refer to the quadrants of a Cartesian plane (if I am standing at the origin and facing along the positive y-axis).
""".strip()

OBJ_REL_DIRECTION_V2_TEMPLATE = """
If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to my left, right, or back?
An object is to my back if I would have to turn at least 135 degrees in order to face it.
""".strip()

OBJ_REL_DIRECTION_V3_TEMPLATE = """
If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to the left or the right of the {orienting_object}?
""".strip()


OBJ_SPTP_DISTANCE_TEMPLATE = """
Which of the these objects ({choice_a}, {choice_b}, {choice_c}, {choice_d}) is the closest to the ego-position at the last frame in the video?
""".strip()


OBJ_APPEARANCE_ORDER_TEMPLATE = """
What will be the first-time appearance order of the following categories in the video: {choice_a}, {choice_b}, {choice_c}, {choice_d}?
""".strip()

ROUTE_PLAN_TEMPLATE = """
You are a robot beginning at the {start_object} facing the {facing_object}. You want to navigate to the {destination_object}. You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'): {actions}
""".strip()

# Camera-to-object absolute distance template
VSTI_CAMERA_OBJ_DIST_TEMPLATE = """
What is the approximate distance (in meters) between the camera (or the person filming) and the nearest point of the {object_name} in {frame_description}?
""".strip()


# Camera-to-object relative distance template
VSTI_CAMERA_OBJ_REL_DIST_TEMPLATE_V1 = """
Measuring from the closest point of each object, which of these objects ({choice_a}, {choice_b}, {choice_c}, {choice_d}) is the closest to the camera in {frame_description}?
""".strip()

VSTI_CAMERA_OBJ_REL_DIST_TEMPLATE_V2 = """
Measuring from the closest point of each object, which of these objects ({choice_a}, {choice_b}, {choice_c}) is the closest to the camera in {frame_description}?
""".strip()

VSTI_CAMERA_OBJ_REL_DIST_TEMPLATE_V3 = """
Measuring from the closest point of each object, which of these objects ({choice_a}, {choice_b}) is the closest to the camera in {frame_description}?
""".strip()

# Object-to-object relative position template
VSTI_OBJ_OBJ_REL_POS_NF_TEMPLATE = """
In {frame_description}, relative to the camera, is {obj_A_name} [Near/Far] compared to {obj_B_name}?
""".strip()

# Object-to-object relative position template
VSTI_OBJ_OBJ_REL_POS_LR_TEMPLATE = """
In {frame_description}, relative to {obj_B_name}, is {obj_A_name} to the [Left/Right]?
""".strip()

# Object-to-object relative position template
VSTI_OBJ_OBJ_REL_POS_UD_TEMPLATE = """
In {frame_description}, relative to {obj_B_name}, is {obj_A_name} to the [Up/Down]?
""".strip()

# Camera motion (translation) template V1 (4 choices)
VSTI_CAMERA_MOVEMENT_DIRECTION_TEMPLATE_V1 = """
During the sequence between {start_frame_description} and {end_frame_description}, what was the primary consistent direction of the camera's movement relative to its orientation at the start? The options are {choice_a}, {choice_b}, {choice_c}, and {choice_d}.
""".strip()

# Camera motion (translation) template V2 (3 choices)
VSTI_CAMERA_MOVEMENT_DIRECTION_TEMPLATE_V2 = """
During the sequence between {start_frame_description} and {end_frame_description}, what was the primary consistent direction of the camera's movement relative to its orientation at the start? The options are {choice_a}, {choice_b}, and {choice_c}.
""".strip()

# Camera motion (translation) template V3 (2 choices)
VSTI_CAMERA_MOVEMENT_DIRECTION_TEMPLATE_V3 = """
During the sequence between {start_frame_description} and {end_frame_description}, what was the primary consistent direction of the camera's movement relative to its orientation at the start? The options are {choice_a} and {choice_b}.
""".strip()

# Camera displacement template
VSTI_CAMERA_DISPLACEMENT_TEMPLATE = """
Approximately how far (in meters) did the camera move between {start_frame_description} and {end_frame_description}?
""".strip()
