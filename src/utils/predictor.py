from tensorflow.keras import models
from urllib.request import Request, urlopen
from PIL import Image
from io import BytesIO
from numpy import array, argmax


# As of now, model is not included in the code
model = models.load_model("keras_model.h5")

# Predict given url
def url_predict(url):

    # Open url and predict
    open_url = urlopen(Request(url = url, headers = {"User-Agent": "Mozilla/5.0"}))
    image = Image.open(BytesIO(open_url.read())).resize((180, 180))
    img_array = (array(image) / 255.0).reshape(1, 180, 180, 3)
    prediction = argmax(model.predict(img_array))

    return prediction
