# import necessary packages
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l2 
from PIL import Image, ImageFile
import base64
from skimage import transform
import numpy as np
import flask
import io 
import wget
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
img_height = 155
img_width = 220

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def create_base_network_signet(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    ''' 
    input = Input(shape=input_shape)
    vgg = vgg16.VGG16(weights="imagenet", include_top=False)

    for layer in vgg.layers:
        layer.trainable = False

    for layer in vgg.layers:
        print(layer, layer.trainable)

    x = Model(inputs=vgg.input,
        outputs=vgg.output)(input)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    return Model(input, x)

def load_model(input_shape):
	global model

	# load the pre-trained keras model
	base_network = create_base_network_signet(input_shape)
	base_network.summary()

	input_a = Input(shape=(input_shape))
	input_b = Input(shape=(input_shape))

	# because we re-use the same instance `base_network`,
	# the weights of the network will be shared across the two branches
	processed_a = base_network(input_a)
	processed_b = base_network(input_b)

	distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

	model_path = 'model.hdf5'
	exists = os.path.isfile(model_path)

	if not exists:
		print('Beginning file download with wget module...')
		url = "https://docs.google.com/uc?export=download&id=1rWeDw9v3qcN6-GCpdEWNu0762vIyuTDm"
		wget.download(url)

	model = Model(inputs=[input_a, input_b], outputs=distance)
	model.load_weights(model_path)
	model._make_predict_function()

def prepare_image(image):
	global img_height
	global img_width

	image = img_to_array(image).astype("float32")
	image= transform.resize(image, (img_height, img_width))
	image *= 1./255
	image = np.expand_dims(image, axis = 0)

	return image

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the view
	data = {"success": False}

	if flask.request.method == "POST":

		body = flask.request.data
		
		if flask.request.data != None:

			body = flask.json.loads(body)

			img1_data = base64.b64decode(str(body['image1']))
			image1 = Image.open(io.BytesIO(img1_data))
			
			# preprocess the image and prepare it
			image1 = prepare_image(image1)

			img2_data = base64.b64decode(str(body['image2']))
			image2 = Image.open(io.BytesIO(img2_data))

			# preprocess the image and prepare it
			image2 = prepare_image(image2)

			pred = model.predict([image1, image2])

			print('Pred Score: {}'.format(pred))

			if pred[0] < 0.5:
				data["success"] = True
			
			data["score"] = str(pred[0])

	return flask.jsonify(data)

print(" Loading keras model and Flask starting server... ")

input_shape = (155, 220, 3)
load_model(input_shape)
