# Train unimodal VAEs to extract relevant information to include in the joint encoder

from pythae.samplers import NormalSampler
from pythae.models import my_VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig, BaseTrainer
from torchvision import datasets
from torchvision.utils import make_grid, save_image
from dataloaders import MNIST_DL
from torch.optim import Adam
import wandb
from os import path
import torch
import os
from models.nn import Encoder_VAE_MNIST,Decoder_AE_MNIST

learning_rate = 1e-3

wandb.init(project = 'vae_mnist', entity="asenellart", config={'lr' : learning_rate}, mode='online') # mode = ['online', 'offline', 'disabled']
wandb.define_metric('epoch')
wandb.define_metric('*', step_metric='epoch')

my_vae_config = model_config = VAEConfig(
    input_dim = (1,28,28),
    latent_dim=20
)

encoder = Encoder_VAE_MNIST(my_vae_config)
decoder = Decoder_AE_MNIST(my_vae_config)

my_vae_model = my_VAE(
    model_config=my_vae_config,
    encoder=encoder,
    decoder=decoder
)
my_vae_model.cuda()

train_dl, eval_dl = MNIST_DL('../data', 'numbers').getDataLoaders(batch_size=256, shuffle=True, device='cuda')


optim = Adam(my_vae_model.parameters(), lr=learning_rate)

output_path = '../experiments/vae_mnist'
if not path.exists(output_path) :
    os.mkdir(output_path)
if not path.exists(output_path + '/samples') :
    os.mkdir(output_path + '/samples')

pipe = NormalSampler(model = my_vae_model)

def train(epoch):
    my_vae_model.train()
    global_loss = 0
    for batch in train_dl:
        data, labels = batch[0].cuda(), batch[1].cuda()
        o = my_vae_model(data)
        loss = o.loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        global_loss += loss
    print(f'===> Epoch {epoch} : train_loss = {global_loss/len(train_dl)}')
    wandb.log({'train_loss' : global_loss/len(train_dl)})
    torch.save(my_vae_model.encoder.state_dict(), output_path + '/encoder.pt')
    torch.save(my_vae_model.decoder.state_dict(), output_path + '/decoder.pt')
    torch.save(my_vae_config, output_path + '/config.pt')

def eval(epoch):
    my_vae_model.eval()
    global_loss = 0
    for batch in eval_dl:
        data, labels = batch[0].cuda(), batch[1].cuda()
        o = my_vae_model(data)
        loss = o.loss
        global_loss += loss
    print(f'===> Epoch {epoch} : test_loss = {global_loss / len(eval_dl)}')
    wandb.log({'test_loss': global_loss / len(eval_dl)})


for epoch in range(20):
    train(epoch)
    eval(epoch)

    if epoch % 5== 0:
        samples = make_grid(pipe.sample(8*8,return_gen=True))
        save_image(samples, output_path + f'/samples/epoch_{epoch}.png')
        wandb.log({'samples' : wandb.Image(samples)})