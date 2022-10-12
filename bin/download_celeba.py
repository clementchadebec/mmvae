'''Download the celeba dataset to the data folder'''


import gdown

url = 'https://drive.google.com/u/0/uc?id=1fm2TXZzCeqxD67dZuSM4G12DWUX_OXtY&export=download'
output = '../data/celeba.zip'
gdown.download(url, output, quiet=False)



import zipfile
with zipfile.ZipFile('../data/celeba.zip', 'r') as zip_ref:
    zip_ref.extractall('../data/')
