# MNIST-SVHN


|Modèle       |   posterior & prior dist    |   decoder distribution |learn prior   |   loss    |  Model specifications| trained | output folder | batchsize | early_stop
|---    |:-:    |:-:    |:-:|:-:    |:-:    |:-: |:-: |:-:| :--
|   JMVAE    |   Normal    |   Normal |False   |JMVAE loss|| no | | 256 | 
|     MMVAE  | Laplace with softmax trick | Normal|False |dreg-looser|   K = 30 |      | |128
|   MVAE    | Normal |  Normal |False| poe |  | no | x| 256
| JMVAE-NF-DCCA | Normal | Normal|False | jmvae-nf | |yes || |256
| JMVAE-NF-DCCA | Normal | Normal|False | jmvae-nf | |yes || |256
|MMVAE with NF| Normal| Normal | False | Elbo |  | no | x | 256
| MoePoe | Normal | Normal |False| moepoe | |no |x 




# CelebA dataset

|Modèle       |   posterior & prior dist    |   decoder distribution    |   loss    |  Model specifications     | trained | output folder |
|---    |:-:    |:-:    |:-:    |:-:    |:-:| :--
|   JMVAE    |   Normal    |   Normal+bce    |JMVAE loss|Trained with 40 epochs warmup| no | x
|     MMVAE  | Laplace with softmax trick | Normal+bce |dreg-looser|   max epochs 100 |    no   |x
|   MVAE    | Normal |  Normal+bce | poe | max epochs 100 | no | x
| MoePoe | Normal | Normal+bce | moepoe | max epochs 100 |no |x 
| JMVAE-NF-DCCA | Normal | Normal+bce | jmvae-nf | warmup 40, max epochs 100 |no |x
| MMVAE-NF| Normal | Normal + bce | elbo | - | no | x 



# Questions

- Dans MMVAE, le prior n'est pas fixe mais appris, faudrait-il que je fasse de même ? 
- Dans l'implémentation originale de JMVAE, le poids de $\beta$ est augmenté au fur et à mesure de l'entraînement : si je veux être parfaitement fidèle il faudrait aussi que je fasse de même 
- Pour la cohérence jointe, j'ai l'impression que dans les expériences précédentes c'était plutôt le prior qui était utilisé pour générer : je devrais peut-être faire pareil ? 