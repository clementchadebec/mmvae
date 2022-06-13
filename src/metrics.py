### Define quantitative metrics to compare models
import numpy as np
import torch
from utils import extract_rayon, negative_entropy




def entropy(model, data, n = 20):
    bdata = [d[:n] for d in data]
    samples = model._sample_from_conditional(bdata, n=100)
    r, range, bins = model.extract_hist_values(samples)
    return negative_entropy(r, range, bins)


