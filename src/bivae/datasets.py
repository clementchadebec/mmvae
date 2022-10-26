import io
import json
import os
import pickle
from collections import Counter, OrderedDict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data import Dataset
from torchvision import transforms, models, datasets


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered."""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class CUBSentences(Dataset):

    def __init__(self, root_data_dir, split, transform=None, **kwargs):
        """split: 'trainval' or 'test' """

        super().__init__()
        self.data_dir = os.path.join(root_data_dir, 'cub')
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 32)
        self.min_occ = kwargs.get('min_occ', 3)
        self.transform = transform
        os.makedirs(os.path.join(root_data_dir, "lang_emb"), exist_ok=True)

        self.gen_dir = os.path.join(self.data_dir, "oc:{}_msl:{}".
                                    format(self.min_occ, self.max_sequence_length))

        if split == 'train':
            self.raw_data_path = os.path.join(self.data_dir, 'text_trainvalclasses.txt')
        elif split == 'test':
            self.raw_data_path = os.path.join(self.data_dir, 'text_testclasses.txt')
        else:
            raise Exception("Only train or test split is available")

        os.makedirs(self.gen_dir, exist_ok=True)
        self.data_file = 'cub.{}.s{}'.format(split, self.max_sequence_length)
        self.vocab_file = 'cub.vocab'

        if not os.path.exists(os.path.join(self.gen_dir, self.data_file)):
            print("Data file not found for {} split at {}. Creating new... (this may take a while)".
                  format(split.upper(), os.path.join(self.gen_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[str(idx)]['idx']
        if self.transform is not None:
            sent = self.transform(sent)
        return sent, self.data[str(idx)]['length']

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _load_data(self, vocab=True):
        with open(os.path.join(self.gen_dir, self.data_file), 'rb') as file:
            self.data = json.load(file)

        if vocab:
            self._load_vocab()

    def _load_vocab(self):
        if not os.path.exists(os.path.join(self.gen_dir, self.vocab_file)):
            self._create_vocab()
        with open(os.path.join(self.gen_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)
        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):
        if self.split == 'train' and not os.path.exists(os.path.join(self.gen_dir, self.vocab_file)):
            self._create_vocab()
        else:
            self._load_vocab()

        with open(self.raw_data_path, 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)

        data = defaultdict(dict)
        pad_count = 0

        for i, line in enumerate(sentences):
            words = word_tokenize(line)

            tok = words[:self.max_sequence_length - 1]
            tok = tok + ['<eos>']
            length = len(tok)
            if self.max_sequence_length > length:
                tok.extend(['<pad>'] * (self.max_sequence_length - length))
                pad_count += 1
            idx = [self.w2i.get(w, self.w2i['<exc>']) for w in tok]

            id = len(data)
            data[id]['tok'] = tok
            data[id]['idx'] = idx
            data[id]['length'] = length

        print("{} out of {} sentences are truncated with max sentence length {}.".
              format(len(sentences) - pad_count, len(sentences), self.max_sequence_length))
        with io.open(os.path.join(self.gen_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."

        with open(self.raw_data_path, 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)

        occ_register = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<exc>', '<pad>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        texts = []
        unq_words = []

        for i, line in enumerate(sentences):
            words = word_tokenize(line)
            occ_register.update(words)
            texts.append(words)

        for w, occ in occ_register.items():
            if occ > self.min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            else:
                unq_words.append(w)

        assert len(w2i) == len(i2w)

        print("Vocablurary of {} keys created, {} words are excluded (occurrence <= {})."
              .format(len(w2i), len(unq_words), self.min_occ))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.gen_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        with open(os.path.join(self.gen_dir, 'cub.unique'), 'wb') as unq_file:
            pickle.dump(np.array(unq_words), unq_file)

        with open(os.path.join(self.gen_dir, 'cub.all'), 'wb') as a_file:
            pickle.dump(occ_register, a_file)

        self._load_vocab()


class CUBImageFt(Dataset):
    def __init__(self, root_data_dir, split, device):
        """split: 'trainval' or 'test' """

        super().__init__()
        self.data_dir = os.path.join(root_data_dir, 'cub')
        self.data_file = os.path.join(self.data_dir, split)
        self.gen_dir = os.path.join(self.data_dir, 'resnet101_2048')
        self.gen_ft_file = os.path.join(self.gen_dir, '{}.ft'.format(split))
        self.gen_data_file = os.path.join(self.gen_dir, '{}.data'.format(split))
        self.split = split

        tx = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        self.dataset = datasets.ImageFolder(self.data_file, transform=tx)

        os.makedirs(self.gen_dir, exist_ok=True)
        if not os.path.exists(self.gen_ft_file):
            print("Data file not found for CUB image features at `{}`. "
                  "Extracting resnet101 features from CUB image dataset... "
                  "(this may take a while)".format(self.gen_ft_file))
            self._create_ft_mat(device)

        else:
            self._load_ft_mat()

    def __len__(self):
        return len(self.ft_mat)

    def __getitem__(self, idx):
        return self.ft_mat[idx]

    def _load_ft_mat(self):
        self.ft_mat = torch.load(self.gen_ft_file)

    def _load_data(self):
        self.data_mat = torch.load(self.gen_data_file)

    def _create_ft_mat(self, device):
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.model = nn.Sequential(*modules)
        self.model.eval()

        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}

        loader = torch.utils.data.DataLoader(self.dataset, batch_size=256,
                                             shuffle=False, **kwargs)
        with torch.no_grad():
            ft_mat = torch.cat([self.model(data[0]).squeeze() for data in loader])

        torch.save(ft_mat, self.gen_ft_file)
        del ft_mat

        data_mat = torch.cat([data[0].squeeze() for data in loader])
        torch.save(data_mat, self.gen_data_file)

        self._load_ft_mat()



############################# MODIFIED CELEBA_DATASET ######################################################*


from functools import partial
import torch
import os
import PIL
from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg


class CelebA(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                ``identity`` (int): label for each person (data points with the same identity are the same person)
                ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                    righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)
            Defaults to ``attr``. If empty, ``None`` will be returned as target.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            len: Union[int, None] = None
    ) -> None:
        import pandas
        super(CelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        self.len = len
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "val": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "val", "test", "all"))]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        # bbox = pandas.read_csv(fn("list_bbox_celeba.csv"), delim_whitespace=True, header=1, index_col=0)
        # landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.csv"), delim_whitespace=True, header=1)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None) if split_ is None else (splits[1] == split_)

        self.filename = splits[mask].index.values
        # self.bbox = torch.as_tensor(bbox[mask].values)
        # self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self) -> None:
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        filename = self.filename[index].split('.')[0] + '.png'
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", "celeba_64x64",'train', filename))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X).float()

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target).float()
        else:
            target = None

        return [(X, target), (target.float().reshape(1,1,40), target)]

    def __len__(self) -> int:
        if self.len is not None:
            return self.len
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

