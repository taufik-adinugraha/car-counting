class TrackableObject:
	def __init__(self, objectID, centroid, target, rect):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
		self.targets = [target]
		self.rects = [rect]

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False