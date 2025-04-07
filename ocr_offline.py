import easyocr
import cv2
# import numpy as np
import matplotlib.pyplot as plt

reader=easyocr.Reader(['en'],gpu=False)
result=reader.readtext("Flipkart_data/maggi.jpg")

print(result)