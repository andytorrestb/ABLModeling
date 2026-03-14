import postprocessing
print(dir(postprocessing))
try:
    print(postprocessing.load_velocity_slices)
except AttributeError as e:
    print(e)