from PIL import Image
import numpy as np

width, height = 500, 500
blank_map = 255 * np.ones((height, width), dtype=np.uint8)  
img = Image.fromarray(blank_map)
img.save('track1.png')
print("Created blank map: track1.png")
