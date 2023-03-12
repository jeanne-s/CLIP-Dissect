import num2words

paths = []
vec = []


class Node:
    def __init__(self, x, name):
        self.data = x
        self.name = name
        self.child = []


def get_path(vec, paths):
    path = ''
    for e in vec:
        path += e + '.'
    paths.append(path)
    return paths


def all_layers_paths(root, paths=[]):

    global vec
    if (not root):
        return
    vec.append(root.name)

    if (len(root.child) == 0): # if leaf node
        paths = get_path(vec, paths)
        vec.pop()
        return
    
    for i in range(len(root.child)):
        all_layers_paths(root.child[i], paths)
        
    vec.pop()
    return paths


def get_route(root):
    global vec
    if (not root):
        return
    paths = all_layers_paths(root)
    paths = [layer[5:-1] for layer in paths]
    return paths


def build_tree_from_net(model):

    root = Node(model, 'root')

    def add_nodes(node, tree, parent_idx):
        for name, layer in node.data.named_children():
            current_node = Node(layer, name)
            tree[parent_idx].child.append(current_node) 
            tree.append(current_node)
            if list(layer.children()) != []: # if not leaf node
                add_nodes(current_node, tree, tree.index(current_node))
        return tree

    nodes_list = add_nodes(root, [root], 0)


    return get_route(nodes_list[0])



# Rename the network's layers which are int 
def rename_layers(obj, old_name, new_name):
    obj._modules[new_name] = obj._modules.pop(old_name)


# Rename ResNet-50 layers 
def rename_resnet50(net): 

    for i in [0,1,2]:
        rename_layers(net.layer1, str(i), num2words.num2words(i)) 
    for i in [0,1,2,3]:
        rename_layers(net.layer2, str(i), num2words.num2words(i)) 
    for i in [0,1,2,3,4,5]:
        rename_layers(net.layer3, str(i), num2words.num2words(i)) 
    for i in [0,1,2]:
        rename_layers(net.layer4, str(i), num2words.num2words(i)) 

    for i in [0, 1]:
        rename_layers(net.layer1.zero.downsample, str(i), num2words.num2words(i))
        rename_layers(net.layer2.zero.downsample, str(i), num2words.num2words(i))
        rename_layers(net.layer3.zero.downsample, str(i), num2words.num2words(i))
        rename_layers(net.layer4.zero.downsample, str(i), num2words.num2words(i))
