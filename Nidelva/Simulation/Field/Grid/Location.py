

# TODO: Replace coordinates with Location as an object
class Location:

    lat = None
    lon = None
    depth = None

    def __init__(self, lat=None, lon=None, depth=None):
        self.lat = lat
        self.lon = lon
        self.depth = depth