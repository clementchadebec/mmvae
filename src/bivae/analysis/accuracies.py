'''

Functions to compute the cross/joint accuracies

'''

import torch




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
                    preds = model.classifiers[j](recon.permute(1,0,2,3,4).resize(len(bdata[0])*ns, *model.shape_mods[j]))
                    labels[i][j] = torch.argmax(preds, dim=1).reshape(len(bdata[0]), ns)
        return labels
    
    
def compute_accuracies(model, data, classes, n_data=20, ns=100):

        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""
        
        
        if (n_data == 'all') or (n_data>len(data[0])):
            n_data = len(data[0])

        # Compute cross_coherence
        labels = conditional_labels(model, data, n_data, ns)
        # Create an extended classes array where each original label is replicated ns times
        classes_mul = torch.stack([classes[0][:n_data] for _ in range(ns)]).permute(1,0).cuda()
        
        accuracies = [[None for _ in range(model.mod)] for _ in range(model.mod)]
        for i in range(model.mod):
            for j in range(model.mod):
                if i!=j:
                    accuracies[i][j] = torch.sum(classes_mul == labels[i][j])/(torch.mul(*classes_mul.size()))
        
        acc_names = [f'acc_{i}_{j}' for i in range(model.mod) for j in range(model.mod) if i!=j]
        acc = [accuracies[i][j] for i in range(model.mod) for j in range(model.mod) if i!=j]
        metrics = dict(zip(acc_names,acc))

        # Compute joint-coherence
        N_joint = ns*n_data
        data = model.generate('', epoch=0, N=N_joint, save=False)
  
        
        metrics['joint_coherence'] = compute_joint_accuracy(model,data)

        return metrics

def compute_joint_accuracy(model, data):
    labels_joint = [torch.argmax(model.classifiers[i](data[i]), dim=1) for i in range(model.mod)]
    pairs_labels = torch.stack([labels_joint[i] == labels_joint[j] for i in range(model.mod) for j in range(model.mod)])
    joint_acc = torch.sum(torch.all(pairs_labels, dim=0))/len(data[0])
    return joint_acc