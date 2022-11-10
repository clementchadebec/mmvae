from torch.hub import load_state_dict_from_url

FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'  # noqa: E501

load_state_dict_from_url(FID_WEIGHTS_URL,'../fid_model/', file_name='model.pt')