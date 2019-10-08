# Normalizing X considering the absolute values of the coordinates
highest_latitude = 22.836006
lowest_latitude = 22.801396
highest_longitude = 47.095658
lowest_longitude = 47.046078
hightest_time = 86400
lowest_time = 0
def adjustTimeColumn(time):
	if time == 0:
		return 0
	return (time - lowest_time)/(hightest_time - lowest_time)

def adjustLatitudeColumn(latitude):
	if latitude == 0:
		return 0
	return (latitude - lowest_latitude)/(highest_latitude - lowest_latitude)

def adjustLongitudeColumn(longitude):
	if longitude == 0:
		return 0
	return (longitude - lowest_longitude)/(highest_longitude - lowest_longitude)

def adjustTrainDf(df):

	for column in df:
		# For time columns
		if column % 3 == 0:
			df[column] = df[column].apply(adjustTimeColumn)
		elif column % 3 == 1:
			df[column] = df[column].apply(adjustLatitudeColumn)
		elif column % 3 == 2:
			df[column] = df[column].apply(adjustLongitudeColumn)

	return df