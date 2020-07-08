#from keras.applications.inception_v3 import (
#    InceptionV3, preprocess_input, decode_predictions)


#from keras.applications.vgg16 import (
#    VGG16, preprocess_input, decode_predictions)

from keras.applications.resnet50 import (
	ResNet50, preprocess_input,decode_predictions)

from keras.models import Model
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import os
from keras.models import load_model
import pickle
from keras.models import model_from_json
import re
import h5py

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def _compute_gradients(tensor, var_list):
     grads = tf.gradients(tensor, var_list)
     return [grad if grad is not None else tf.zeros_like(var)
     for var, grad in zip(var_list, grads)]


def grad_cam(input_model, image, category_index, layer_name,nb_classes,dims):
    #nb_classes = 2
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
    model.summary()
    loss = K.sum(model.output)
    print('loss:')
    print(loss)
    conv_output =  [l for l in model.layers if l.name is layer_name][0].output
    #conv_output =  input_model.layers[299].output
    print('model output shape:')
    print(conv_output.get_shape().as_list())
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([input_model.input], [conv_output, grads])
    print('grads:')
    print(grads)
    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    print(list(output)[0])
    print(len(list(output)))
    print(list(grads_val))
    weights = np.mean(grads_val, axis = (0, 1))
    print("weights vector")
    print(weights)
    print(np.mean(weights))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)
    print(output.shape)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (dims, dims))
    cam = np.maximum(cam, 0)
    print(np.max(cam))
    print(np.mean(cam))
    heatmap = cam / np.max(cam)
    #heatmap = cam / np.min(cam) 
     
    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    print(np.min(image))
    image -= np.min(image)
    image = np.minimum(image, 255)
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_PARULA)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap



def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path,dims):
    img = image.load_img(path, target_size=(dims, dims))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x




def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    #layer_output = model.layers[299].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])


def modify_backprop(model, name,nm):
	g = tf.get_default_graph()
	with g.gradient_override_map({'Relu': name}):

        	# get layers that have an activation
		layer_dict = [layer for layer in model.layers[1:]
			if hasattr(layer, 'activation')]

        	# replace relu activation
		for layer in layer_dict:
			if layer.activation == keras.activations.relu:
				layer.activation = tf.nn.relu

		# re-instantiate a new model
		#new_model = VGG16(weights=None,include_top=True,classes=2)
		new_model = ResNet50(weights=None,include_top=True,classes=2)
		new_model.load_weights(nm)

	return new_model



def load_image(path,dims):
    img = image.load_img(path, target_size=(dims, dims))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x



#trained_weights="/gpfs/data/abl/deepomics/tsirigoslab/melanoma/BRAF_classifier_data/Training/inceptionV3_Output/weights-improvement-01-0.63.hdf5"
#trained_weights="/gpfs/data/abl/deepomics/tsirigoslab/melanoma/BRAF_classifier_data/Training/inceptionV3_Output/transfer_learning/weights-improvement-02-0.61.hdf5"
#trained_weights="/gpfs/data/abl/deepomics/tsirigoslab/melanoma/BRAF_classifier_data/Training/inceptionV3_Output/transfer_learning/V2/weights-improvement-01-0.61.hdf5"
#trained_weights="/gpfs/data/abl/deepomics/tsirigoslab/melanoma/BRAF_classifier_data/Training/VGG16_Output/weights-improvement-03-0.64.hdf5"
trained_weights="/gpfs/data/abl/deepomics/tsirigoslab/melanoma/BRAF_classifier_data/Training/ResNet50_Output/weights-improvement-02-0.62.hdf5"

#output_weights='Inceptionv3_tl2_savedWeights.hd5'
#output_weights='VGG16_BRAF_WT_savedWeights.hd5'
output_weights='ResNet50_savedWeights.hd5'

model=load_model(trained_weights)
model=model.layers[-2]
model.save_weights(output_weights)

#model = InceptionV3(weights=None,include_top=True,classes=2)
#model = VGG16(weights=None,include_top=True,classes=2)
model = ResNet50(weights=None,include_top=True,classes=2)

model.load_weights(output_weights)
print(model.summary())
model_dims=int(str(model.layers[0].output.shape)[3:7].strip())
print("Model Input Dimensions:"+str(model_dims))
#partition_file='/gpfs/data/abl/deepomics/tsirigoslab/melanoma/BRAF_classifier_data/20x/CreatingInputDictionaries/InceptionV3_topSplit_partition.pickle'
partition_file="/gpfs/data/abl/deepomics/tsirigoslab/melanoma/BRAF_classifier_data/VGG16_20x/CreatingInputDictionaries/VGG16_topSplit_partition.pickle"

with open(partition_file ,'rb') as handle:
       partition = pickle.load(handle)

img_base_dir='/gpfs/scratch/ee699/WSI_preprocessing_melanoma/VGG16_data_20x/'
#img_base_dir='/gpfs/scratch/ee699/WSI_preprocessing_melanoma/InceptionV3_data/'

#random_selection=np.random.choice(len(partition['test']),10)
#random_selection=[54143, 33526, 31071,  2762, 33698, 34218, 66810, 66691, 34301 ,59009]
random_selection=[34939,34940  ,2762, 33698]
print("randomly selected indexes: "+str(random_selection))

backprop_o='guided_backprop_resnet50/'
grad_cam_o='grad_cams_resnet50/'

for selection in random_selection:
    print("selected image:"+str(partition['test'][selection]))
    img=load_image(img_base_dir + partition['test'][selection],dims=model_dims)
    predictions=model.predict(img)
    print(predictions[0][0])
    print(predictions[0][1])


    predicted_class = np.argmax(predictions)
    print('Predicted class:')
    print(predicted_class)

	
    #cam, heatmap = grad_cam(model, img, predicted_class, "conv2d_94",2)
    #cam, heatmap = grad_cam(model, img, predicted_class, "block5_conv3",2,dims=model_dims)
    cam, heatmap = grad_cam(model, img, predicted_class, "res5c_branch2c",2,dims=model_dims)
    print("Success")
    tile='grad_cam_test/'+partition['test'][selection]
    tile_=re.split('/',tile)[len(re.split('/',tile))-1]

    print("grad_cam_test/"+partition['test'][0])
    print(tile_)
    #cam2 = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_PARULA)
    # to overlay
    #alpha = 0.5
    #over = cv2.addWeighted(img, alpha, cam2, 1-alpha, 0)
    cv2.imwrite(grad_cam_o+tile_, cam)


    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp',nm=output_weights)
    saliency_fn = compile_saliency_function(guided_model)
    saliency = saliency_fn([img, 0])
    gradcam = saliency[0] * heatmap[..., np.newaxis]
    cv2.imwrite(backprop_o+tile_, deprocess_image(gradcam))
    print("sucess")
