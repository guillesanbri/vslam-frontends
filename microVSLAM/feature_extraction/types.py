class Keypoint:
    def __init__(self, x, y, strength=None, angle=None):
        self.x = x
        self.y = y
        self.strength = strength
        self.angle = angle


class Descriptor:
    def __init__(self, bits):
        self.bits = bits


class Feature:
    def __init__(self, keypoint, descriptor):
        self.keypoint = keypoint
        self.descriptor = descriptor
