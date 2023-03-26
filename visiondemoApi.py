import os, io
from google.cloud import vision
from google.cloud.vision_v1 import types
import pandas as pd

client = vision.ImageAnnotatorClient()

image_path = f'C:/Users/0007HA744/env/ups-label-package.jpg'

with io.open(image_path, 'rb') as image_file:
    content = image_file.read()

# construct an iamge instance
image = vision.Image(content=content)

"""
# or we can pass the image url
image = vision.types.Image()
image.source.image_uri = 'https://edu.pngfacts.com/uploads/1/1/3/2/11320972/grade-10-english_orig.png'
"""

# annotate Image Response
response = client.text_detection(image=image)  # returns TextAnnotation
df = pd.DataFrame(columns=['locale', 'description'])

texts = response.text_annotations
for text in texts:
    df = df.append(
        dict(
            locale=text.locale,
            description=text.description
        ),
        ignore_index=True
    )

print(df['description'][0])
