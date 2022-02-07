import numpy as np

# ==== Field Config ====
DEPTH = [.5, 1, 1.5, 2.0, 2.5]
DISTANCE_LATERAL = 120
DISTANCE_VERTICAL = np.abs(DEPTH[1] - DEPTH[0])
DISTANCE_TOLERANCE = 1
DISTANCE_SELF = 20
THRESHOLD = 28
# ==== End Field Config ====

# ==== GP Config ====
SILL = .5 # 0.5
RANGE_LATERAL = 550
RANGE_VERTICAL = 2
NUGGET = .04
# ==== End GP Config ====

# ==== Plot Config ======
VMIN = 16
VMAX = 28
# ==== End Plot Config ==

