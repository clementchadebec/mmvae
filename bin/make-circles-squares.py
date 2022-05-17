# Make a paired circles and squares toy dataset for multimodal encoding
import os

import numpy as np
from PIL import Image
import torch
from sklearn.model_selection import train_test_split

dataset_size = 10000
size_image = 32
min_rayon, max_rayon = 0.3, 0.9
circle_thickness = 0.1
n_repeat = 10
output_path = '../data/circles_squares'
if not os.path.exists(output_path):
    os.mkdir(output_path)

rayons = np.linspace(min_rayon,max_rayon,dataset_size)
x = np.linspace(-1,1,size_image)

def circle(X,Y,r):
    return (X**2 + Y**2 <= (r + circle_thickness/2)**2)*(X**2 + Y**2 >= (r - circle_thickness/2)**2)

def square_line(X,Y,r):
    return (np.abs(X) + np.abs(Y) <= (r + circle_thickness/2))*(np.abs(X) + np.abs(Y) >= r - circle_thickness/2)

squares = []
circles = []
labels = []

for i, r_disc in enumerate(rayons):
    for _ in range(n_repeat):
        X,Y = np.meshgrid(x,x)

        # Associate a random-sized disc to a random-sized full square
        img_full_disc = X**2 + Y**2 <= np.random.uniform(min_rayon,max_rayon)**2
        img_full_square = np.abs(X) + np.abs(Y) <= np.random.uniform(min_rayon,max_rayon)
        # And a random-sized ring to a random sized line-square
        img_empty_disc = circle(X,Y,np.random.uniform(min_rayon,max_rayon))
        img_empty_square = square_line(X,Y,np.random.uniform(min_rayon,max_rayon))

        squares.extend([img_full_square, img_empty_square])
        circles.extend([img_full_disc, img_empty_disc])
        labels.extend([1,0])


# Visualize some examples
output_examples = output_path + '/examples'
if not os.path.exists(output_examples):
    os.mkdir(output_examples)

for i in np.linspace(0,dataset_size*n_repeat-1, 100):
    i = int(i)
    img = Image.fromarray(np.concatenate([squares[i], circles[i]]))
    img.save(output_examples + f'/example_{i}.png')



# Save in pytorch format
squares = torch.unsqueeze(torch.FloatTensor(squares), 1)
circles = torch.unsqueeze(torch.FloatTensor(circles), 1)
labels = torch.tensor(labels)
# Select some for training and testing
s_train, s_test, c_train, c_test, l_train, l_test = train_test_split(squares,circles, labels, test_size=0.3)

torch.save(s_train, output_path + '/circles_train.pt')
torch.save(s_test, output_path + '/circles_test.pt')
torch.save(c_train, output_path + '/discs_train.pt')
torch.save(c_test, output_path + '/discs_test.pt')
torch.save(l_train, output_path+ '/labels_train.pt')
torch.save(l_test, output_path + '/labels_test.pt')

print(c_train.shape, c_test.shape)




