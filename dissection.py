import numpy as np
import torch
import torch.nn as nn
from clip_main import clip 
from tqdm import tqdm
import matplotlib.pyplot as plt

import utils



def preprocess_concept_set(set_file):
    """ Returns a list of the words in set_file.
    """
    set = open(set_file, "r").read()
    txt_into_list = set.split("\n")
    return txt_into_list


# --- STEP 1
def encode_concept(S, m, model, device):
    """ Computes the text embedding of the m-th concept in the concept set S
    using the text encoder of a CLIP model. Correspond to T_m.
    """
    token = clip.tokenize(S[m], context_length=77).to(device)
    with torch.no_grad():
        T_m = model.encode_text(token)
    
    return T_m
    

def encode_image(D, i, model, preprocess, device):
    """ Computes the image embedding of the i-th image in the probing dataset
    D using the image encoder of a CLIP model. Corresponds to I_i.
    """
    image, _ = D[i]
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        I_i = model.encode_image(image_input)
    
    return I_i


def vector_concept_activation(text_embedding, image_embedding): 
    """ Computes the vector of concept-activation, which is the inner 
    product of a text embedding and an image embedding.
    """
    text = text_embedding.numpy()[0]
    ima = image_embedding.numpy()[0]

    return np.dot(text, ima)


# --- STEP 2
def select_conv_layers(model):
    """ Returns a list of all Conv2d layers' names in the model.
    """
    all_layers_names = utils.build_tree_from_net(model)

    # Among all layers, select conv layer only (ONLY FOR RESNET-50 !)
    all_layers_names = [layer for layer in all_layers_names if (layer[-5:-1]=='conv' or layer[-4:]=='zero')]
    print("All conv layers :", all_layers_names)
    return all_layers_names


def neuron_activation(x, k, model, all_layers_names):
    """ Computes the activation of the k-th neuron for image x.
    Corresponds to A_k(x_i).
    """
    activation = {}
    def getActivation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    layer_name = all_layers_names[k].split('.')

    # Register forward hooks on the layers of choice
    attr = model
    for i in range(0, len(layer_name)-1):
        attr = getattr(attr, layer_name[i])
    h1 = attr._modules[layer_name[-1]].register_forward_hook(getActivation('conv_layer'))
    out = model(x)
    h1.remove()
    return activation.get('conv_layer')


def g_mean_function(activation_map, k):
    """ Computes the mean of the activation map over spatial dimensions.
    Corresponds to g(A_k(x_i)).
    """
    return torch.mean(activation_map)


def plot_highest_activating_images(q, probing_dataset, label, channel, nb_images=5):
    idx_images = []
    idx_images = np.argsort(q)[:nb_images]

    print(idx_images)
    # Subplots size nb_images
    for i in range(0, nb_images):
        plt.subplot(1, 5, i+1)
        plt.imshow(probing_dataset[idx_images[i]][0])
    plt.suptitle("Highest activating images \nNeuron label and channel : {}, {}".format(label, channel))
    plt.show()
    return 


def neuron_label(k, concept_set, probing_dataset, probed_network, preprocess_probed, model, preprocess, M, N, device, ord=2):
    """ Returns the label of neuron k, defined as the concept which 
    minimizes the distance between its concept-image vector and k-th
    neuron activation vector. The distance is the l-norm of order p.
    """

    N = N if N else len(probing_dataset)
    M = M if M else len(concept_set)

    concept_set = preprocess_concept_set(concept_set)

    I = []
    for i in tqdm(range(0, N), desc="Image Encoding"):
        I.append(encode_image(probing_dataset, i, model, preprocess, device))
    I = torch.cat((I), 0)

    T = []
    for m in tqdm(range(0, M), desc="Text Encoding"):
        T.append(encode_concept(concept_set, m, model, device))
    T = torch.cat((T), 0)

    # Vector of concept-activation
    p = np.dot(np.array(I), np.transpose(np.array(T)))

    # Rename layers in ResNet-50 which are int
    utils.rename_resnet50(probed_network)

    # Activation vector
    layers_names = select_conv_layers(probed_network)
    channel = 0
    nb_channels = neuron_activation(preprocess_probed(probing_dataset[i][0]).unsqueeze(0), k, probed_network, layers_names).shape[1] -1
    channel = int(input("Select channel for layer {} in range [0, {}] : ".format(layers_names[k], nb_channels)))
    print("Selected layer and channel: {}, {}".format(layers_names[k], channel))
    q_k = [g_mean_function(neuron_activation(preprocess_probed(probing_dataset[i][0]).unsqueeze(0), k, probed_network, layers_names)[0,channel,:,:], k) for i in tqdm(range(0, N), desc="Neuron Activation")] 

    # Distance
    d_mk = np.linalg.norm(np.transpose(np.array(p)) - np.array(q_k), axis=0, ord=ord)

    # Argmin of the distance
    l = np.argmin(d_mk)

    # Label
    label = concept_set[l]

    # Highest activating images
    plot_highest_activating_images(q_k, probing_dataset, label, channel)

    return label