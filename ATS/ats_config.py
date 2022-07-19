is_radius_th = False
radius_th = 0.00001  # If the distance of each point is less than this value, it is considered that there is no coverage

boundary = 0  # Lower bound
up_boundary = 0.99  # upper bound

is_log = False  # Whether the distance function uses log
linear_ratio = 0.005  # Coverage radius of each point ratio *d   linear
log_ratio = 0.01  # log

round_num = 5  # Floating point number constraint is applied to the endpoint, with a maximum of 5 bits
