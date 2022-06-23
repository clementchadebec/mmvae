# Make a paired circles and squares toy dataset for multimodal encoding
# In this example dataset : on of the modalities have a random color grey background dataset that is modality specific information :
# The circles all have the same size and the shared information is the shape

import os

import numpy as np
from PIL import Image
import torch
from sklearn.model_selection import train_test_split

dataset_size = 5000
size_image = 32
min_background, max_background = 5, 250
min_pertub, max_pertub = 0,0.3
rayon = r = 0.5
circle_thickness = 0.2
output_path = '../data/empty_full_bk'
if not os.path.exists(output_path):
    os.mkdir(output_path)

x = np.linspace(-1,1,size_image)

def circle(X,Y):
    c = np.random.uniform(-max_pertub, max_pertub, 2)
    shape = ((X-c[0])**2 + (Y-c[1])**2 <= (r + circle_thickness/2)**2)*((X-c[0])**2 + (Y-c[1])**2 >= (r - circle_thickness/2)**2)
    bk_color = np.random.randint(min_background, max_background)/255
    bk = (1-shape)*bk_color
    return shape + bk, bk_color

def square_line(X,Y):
    c = np.random.uniform(-max_pertub, max_pertub, 2)
    shape = (np.abs(X-c[0]) + np.abs(Y-c[1]) <= (r + circle_thickness/2))*(np.abs(X-c[0]) + np.abs(Y-c[1]) >= r - circle_thickness/2)
    bk_color = np.random.randint(min_background, max_background)/255
    bk = (1-shape)*bk_color
    return shape + bk, bk_color

empty, background , full = [], [], []

labels = []

for i in range(dataset_size):
    X, Y = np.meshgrid(x, x)
    # Add a circle pair to the dataset
    c, bkc = circle(X,Y)
    empty.append(c)
    background.append(bkc)
    full.append(X**2 + Y**2 <= r**2)
    # And then a square one
    c, bkc = square_line(X, Y)
    empty.append(c)
    background.append(bkc)
    full.append(np.abs(X) + np.abs(Y) <= r)
    labels.extend([0,1])




# Visualize some examples
output_examples = output_path + '/examples'
if not os.path.exists(output_examples):
    os.mkdir(output_examples)

for i in np.linspace(0,dataset_size-1, 100):
    i = int(i)
    img = Image.fromarray(np.uint8(np.concatenate([empty[i]*255, full[i]*255])))
    img.save(output_examples + f'/example_{i}.png')



# Save in pytorch format
empty = torch.unsqueeze(torch.FloatTensor(empty), 1)
full = torch.unsqueeze(torch.FloatTensor(full), 1)
labels, background = torch.tensor(labels),torch.tensor(background)
# Select some for training and testing
s_train, s_test, c_train, c_test, idx_train, idx_test = train_test_split(empty,full, np.arange(len(labels)), test_size=0.3)


torch.save(s_train, output_path + '/squares_train.pt')
torch.save(s_test, output_path + '/squares_test.pt')
torch.save(c_train, output_path + '/circles_train.pt')
torch.save(c_test, output_path + '/circles_test.pt')
torch.save(labels[idx_train], output_path+ '/labels_train.pt')
torch.save(labels[idx_test], output_path + '/labels_test.pt')
torch.save(background[idx_train], output_path + '/r_squares_train.pt')
torch.save(background[idx_test], output_path + '/r_squares_test.pt')
torch.save(background[idx_train], output_path +'/r_circles_train.pt')
torch.save(background[idx_test], output_path +'/r_circles_test.pt')

print(c_train.shape, c_test.shape)




