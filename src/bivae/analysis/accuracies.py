'''

Functions to compute the cross/joint accuracies

'''

import torch


def conditional_labels_(model,classifier1,classifier2, data, n_data=8, ns=30):
    """ Sample ns from the conditional distribution (for each of the first n_data)
    and compute the labels in this conditional distribution (based on the
    predefined classifiers)
    
    only suitable for 2 modalities
    """

    bdata = [d[:n_data] for d in data]
    samples = model._sample_from_conditional(bdata, n=ns)
    cross_samples = [torch.stack(samples[0][1]), torch.stack(samples[1][0])]

    # Compute the labels
    preds2 = classifier2(cross_samples[0].permute(1, 0, 2, 3, 4).resize(n_data * ns, *model.shape_mod2))  # 8*n x 10
    labels2 = torch.argmax(preds2, dim=1).reshape(n_data, ns)

    preds1 = classifier1(cross_samples[1].permute(1, 0, 2, 3, 4).resize(n_data * ns, *model.shape_mod1))  # 8*n x 10
    labels1 = torch.argmax(preds1, dim=1).reshape(n_data, ns)

    return labels2, labels1


def compute_accuracies_(model, classifier1, classifier2, data, classes, n_data=20, ns=100):

    """ Given the data, we sample from the conditional distribution and compute conditional
    accuracies. We also sample from the joint distribution of the model and compute
    joint accuracy
    
    Only suitable for two modalities"""

    # Compute cross_coherence
    labels2, labels1 = conditional_labels_(model,classifier1,classifier2,data, n_data, ns)

    # Create an extended classes array where each original label is replicated ns times
    classes_mul = torch.stack([classes[0][:n_data] for _ in range(ns)]).permute(1,0).cuda()
    acc2 = torch.sum(classes_mul == labels2)/(n_data*ns)
    acc1 = torch.sum(classes_mul == labels1)/(n_data*ns)

    metrics = dict(accuracy1 = acc1, accuracy2 = acc2)
    data = model.generate('', 0, N=ns, save=False)
    labels_mnist = torch.argmax(classifier1(data[0]), dim=1)
    labels_svhn = torch.argmax(classifier2(data[1]), dim=1)

    joint_acc = torch.sum(labels_mnist == labels_svhn) / ns
    metrics['joint_coherence'] = joint_acc

    return metrics


def conditional_labels(model, data, n_data=8, ns=30):
        """ Sample ns from the conditional distribution (for each of the first n_data)
        and compute the labels in this conditional distribution (based on the
        predefined classifiers)"""

        bdata = [d[:n_data] for d in data]
        samples = model._sample_from_conditional( bdata, n=ns)
        labels = [[None for _ in range(model.mod)] for _ in range(model.mod) ]
        for i in range(model.mod):
            for j in range(model.mod):
                if i!=j:
                    recon = torch.stack(samples[i][j])
                    preds = model.classifiers[j](recon.permute(1,0,2,3,4).resize(n_data*ns, *model.shape_mods[j]))
                    labels[i][j] = torch.argmax(preds, dim=1).reshape(n_data, ns)
        return labels
    
    
def compute_accuracies(model, data, classes, n_data=20, ns=100):

        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""


        # Compute cross_coherence
        labels = conditional_labels(model, data, n_data, ns)
        # Create an extended classes array where each original label is replicated ns times
        classes_mul = torch.stack([classes[0][:n_data] for _ in range(ns)]).permute(1,0).cuda()
        
        accuracies = [[None for _ in range(model.mod)] for _ in range(model.mod)]
        for i in range(model.mod):
            for j in range(model.mod):
                if i!=j:
                    accuracies[i][j] = torch.sum(classes_mul == labels[i][j])/(n_data*ns)
        
        acc_names = [f'acc_{i}_{j}' for i in range(model.mod) for j in range(model.mod) if i!=j]
        acc = [accuracies[i][j] for i in range(model.mod) for j in range(model.mod) if i!=j]
        metrics = dict(zip(acc_names,acc))

        # Compute joint-coherence
        data = model.generate('', epoch=0, N=ns, save=False)
        labels_joint = [torch.argmax(model.classifiers[i](data[i]), dim=1) for i in range(model.mod)]
        
        pairs_labels = torch.stack([labels_joint[i] == labels_joint[j] for i in range(model.mod) for j in range(model.mod)])
        joint_acc = torch.sum(torch.all(pairs_labels, dim=0))/(ns)
        metrics['joint_coherence'] = joint_acc

        return metrics
