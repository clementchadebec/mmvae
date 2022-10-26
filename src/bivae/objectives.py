
# objectives of choice
import torch
from numpy import prod
import torch.nn.functional as F
from utils import log_mean_exp, is_multidata, kl_divergence, wasserstein_2, update_details


# helper to vectorise computation
def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)


def elbo(model, x, K=1, beta_prior = 1):
    """Computes E_{p(x)}[ELBO] """
    qz_x, px_z, _ = model(x)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.llik_scaling
    kld = kl_divergence(qz_x, model.pz(*model.pz_params))
    return (lpx_z.sum(-1) - beta_prior * kld.sum(-1)).mean(0).sum()


def _iwae(model, x, K):
    """IWAE estimate for log p_\theta(x) -- fully vectorised."""
    qz_x, px_z, zs = model(x, K)
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.llik_scaling
    lqz_x = qz_x.log_prob(zs).sum(-1)
    return lpz + lpx_z.sum(-1) - lqz_x


def iwae(model, x, K):
    """Computes an importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    """
    S = compute_microbatch_split(x, K)
    lw = torch.cat([_iwae(model, _x, K) for _x in x.split(S)], 1)  # concat on batch
    return log_mean_exp(lw).sum()


def _dreg(model, x, K):
    """DREG estimate for log p_\theta(x) -- fully vectorised."""
    _, px_z, zs = model(x, K)
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.llik_scaling
    qz_x = model.qz_x(*[p.detach() for p in model.qz_x_params])  # stop-grad for \phi
    lqz_x = qz_x.log_prob(zs).sum(-1)
    lw = lpz + lpx_z.sum(-1) - lqz_x
    return lw, zs


def dreg(model, x, K, regs=None):
    """Computes a doubly-reparameterised importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    """
    S = compute_microbatch_split(x, K)
    lw, zs = zip(*[_dreg(model, _x, K) for _x in x.split(S)])
    lw = torch.cat(lw, 1)  # concat on batch
    zs = torch.cat(zs, 1)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 0, keepdim=True)).exp()
        if zs.requires_grad:
            zs.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).sum()


# multi-modal variants
def m_elbo_naive(model, x, K, epoch, warmup, beta_prior):
    """Computes E_{p(x)}[ELBO] for multi-modal vae --- NOT EXPOSED"""
    qz_xs, px_zs, zss, qz_x_params = model(x)
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        klds.append(kld.sum(-1))
        for d, px_z in enumerate(px_zs[r]):
            lpx_z = px_z.log_prob(x[d]) * model.lik_scaling[d]
            lpx_zs.append(lpx_z.view(*px_z.batch_shape[:2], -1).sum(-1))
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    return obj.mean(0).sum(), {}


def m_elbo(model, x, K=1, beta=1000, epoch=1, warmup=0,beta_prior = 1):
    """Computes importance-sampled m_elbo (in notes3) for multi-modal vae

    Personal comment : I actually don't understand where in this function is computed the
    log(q(z_i|x1;m) --> it feels like it s missing
    """
    qz_xs, px_zs, zss,_ = model(x, K=K)
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        klds.append(kld.sum(-1))
        for d in range(len(px_zs)):
            lpx_z = px_zs[d][d].log_prob(x[d]).view(*px_zs[d][d].batch_shape[:2], -1)
            lpx_z = (lpx_z * model.lik_scaling[d]).sum(-1)
            if d == r:
                lwt = torch.tensor(0.0)
            else:
                zs = zss[d].detach() # the detach implements the stop-grad
                lwt = (qz_x.log_prob(zs) - qz_xs[d].log_prob(zs).detach()).sum(-1)
            lpx_zs.append(lwt.exp() * lpx_z)
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))

    details = dict(lpx_zs00 = lpx_zs[0].sum(), lpx_zs01  = lpx_zs[1].sum(),
                   lpx_zs10 = lpx_zs[2].sum(), lpx_zs11 = lpx_zs[3].sum())
    return obj.mean(0).sum(), details





def _m_iwae(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs): # enumerate on modalities
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1) # sum on latent dimension (the covariance is diagonal)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs])) # compute the mmvae joint posterior
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    lws = torch.cat(lws)
    return lws # (n_modality * K_iwae) x batch_size

def _m_vaevae(model, x, dist, K=1, beta=1000, epoch = 1, warmup = 0, beta_prior=1):
    """ We train the vaes maximizing the unimodal ELBO for each modality with an additionnal term
    that regularizes the distance between posteriors of each modality KL(q(z|x1) || q(z|x2) )
    only for two modalities"""
    loss1 = elbo(model.vaes[0], x[0], beta_prior=beta_prior)
    loss2 = elbo(model.vaes[1], x[1], beta_prior=beta_prior)

    qz_x0, px0_z,_ = model.vaes[0](x[0])
    qz_x1, px1_z, _ = model.vaes[1](x[1])
    if model.align != -1:
        reg = 1/2*(dist(qz_x0,qz_x1)[:,:model.align].mean(0).sum(-1) + dist(qz_x1,qz_x0)[:,:model.align].mean(0).sum()) # symetric distance
    else :
        reg = 1/2*(dist(qz_x0,qz_x1).mean(0).sum(-1) + dist(qz_x1,qz_x0).mean(0).sum()) # symetric distance

    details = dict(loss = loss1 + loss2 , reg = reg, loss1 = loss1, loss2 = loss2)

    return (loss1 + loss2 - beta*reg, details) if epoch >= warmup else (loss1 + loss2, details)

def m_vaevae_kl(model, x, K=1, beta=1000, epoch=1, warmup=0, beta_prior = 1):
    return _m_vaevae(model, x, kl_divergence, K, beta,epoch, warmup, beta_prior)

def m_vaevae_w2(model, x, K=1, beta=1000, epoch=1, warmup=0, beta_prior = 1):
    return _m_vaevae(model, x, wasserstein_2, K, beta, epoch, warmup, beta_prior)

def m_jmvae(model, x, K=1, beta=0, epoch=1, warmup=0, beta_prior = 1):
    """Computes jmvae loss"""
    if not hasattr(model, 'joint_encoder'):
        raise TypeError('The model must have a joint encoder for this loss.')
    if epoch >= warmup:
        model.joint_encoder.requires_grad_(False)
    qz_xy, pxy_z, z_xy = model.forward_joint(x,K=1)
    qz_xs, px_zs, z_xs = model.forward(x, K=1)
    loss, details = 0, {}
    for m,px_z in enumerate(pxy_z):
        loss+= px_z.log_prob(x[m]).squeeze().mean(0).sum()
    # Joint ELBO
    loss = loss - beta_prior*kl_divergence(qz_xy,model.pz(*model.pz_params)).mean(0).sum()
    # KL regularizers
    kl1 = kl_divergence(qz_xy, qz_xs[0]).mean(0).sum()
    kl2 = kl_divergence(qz_xy,qz_xs[1]).mean(0).sum()
    details['kl1'], details['kl2'] , details['loss'] = kl1,kl2, loss
    return (loss - beta*(kl1+kl2), details) if epoch >= warmup else (loss, details)


recon_loss_dict = {'mse' : F.mse_loss, 'bce' : F.binary_cross_entropy , 'l1' : F.l1_loss}

def m_jmvae_nf(model,x,K=1, epoch=1, warmup=0, beta_prior=1):
    if epoch >= warmup:
        model.joint_encoder.requires_grad_(not model.fix_jencoder) #fix the joint encoder
        for vae in model.vaes:
            vae.decoder.requires_grad_(not model.fix_decoders) #fix the decoders
    qz_xy, recons, z_xy = model.forward(x)
    # mu, std = model.joint_encoder.forward(x)
    loss, details = 0, {}
    for m, xm in enumerate(x):
        assert recons[m].shape == xm.shape , f'Sizes are different : {recons[m].shape,xm.shape}'

        F_loss = recon_loss_dict[model.recon_losses[m]]

        details[f'loss_{m}'] = F_loss(
                recons[m].reshape(xm.shape[0], -1),
                xm.reshape(xm.shape[0], -1),
                reduction="none",
            ).sum()*model.lik_scaling[m]

        loss = loss - details[f'loss_{m}']


    details['loss'] = loss
    # KLD to the prior
    mu, log_var = qz_xy.mean, 2*torch.log(qz_xy.stddev)
    # print(list(model.joint_encoder.parameters())[0].requires_grad)
    details['kld_prior'] = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).sum()
    # Approximate the posterior
    if epoch >= warmup:
        reg, det = model.compute_kld(x)
        details['reg'] = reg
        update_details(details, det)
    else :
        details['reg']=0

    return (- details['reg'], details) if epoch >= warmup \
        else (loss - beta_prior*details['kld_prior'], details)

def m_jmvaegan_nf(model,x,K=1, epoch=1, warmup=0, beta_prior=1):
    if epoch >= warmup:
        model.joint_encoder.requires_grad_(not model.fix_jencoder) #fix the joint encoder
        for vae in model.vaes:
            vae.decoder.requires_grad_(not model.fix_decoders) #fix the decoders
    qz_xy, recons, z_xy = model.forward(x)
    # mu, std = model.joint_encoder.forward(x)
    encoder_loss, decoder_loss, discriminator_loss, details = 0, 0, 0, {}
    for m, xm in enumerate(x):
        assert recons[m].shape == xm.shape , f'Sizes are different : {recons[m].shape,xm.shape}'

        if m == 0:
            N = z_xy.shape[0]  # batch size

            z_prior = torch.randn_like(z_xy, device=xm.device)#.requires_grad_(True)

            recon_loss = 0

            for recon_lay in model.reconstruction_layer:
                

                # feature maps of true data
                true_discr_layer = model.discriminator(
                    xm, output_layer_levels=[recon_lay]
                )[f"embedding_layer_{recon_lay}"]

                # feature maps of recon data
                recon_discr_layer = model.discriminator(
                    recons[0], output_layer_levels=[recon_lay]
                )[f"embedding_layer_{recon_lay}"]

  
                # MSE in feature space for images
                recon_loss += F.mse_loss(
                    true_discr_layer.reshape(N, -1),
                    recon_discr_layer.reshape(N, -1),
                    reduction="none",
                ).sum(dim=-1).mean(dim=0)

            encoder_loss = encoder_loss - recon_loss

            gen_prior = model.vaes[0].decoder(z_prior).reconstruction

            true_adversarial_score = torch.sigmoid(model.discriminator(xm).embedding.flatten())
            # gen_adversarial_score = self.discriminator(recon_x).embedding.flatten()
            prior_adversarial_score = torch.sigmoid(model.discriminator(gen_prior).embedding.flatten())

            true_labels = torch.ones(N, requires_grad=False).to(xm.device)
            fake_labels = torch.zeros(N, requires_grad=False).to(xm.device)

            original_dis_cost = F.binary_cross_entropy(
                true_adversarial_score, true_labels
            )  # original are true
            prior_dis_cost = F.binary_cross_entropy(
                prior_adversarial_score, fake_labels
            )#.sum()  # prior is false
            # gen_cost =  F.binary_cross_entropy(
            #   gen_adversarial_score, fake_labels
            # ) # generated are false

            discriminator_loss = discriminator_loss - (
                (original_dis_cost)
                + (prior_dis_cost)
                # +
                # (gen_cost)
            ).mean(dim=0)

            decoder_loss = decoder_loss - (
                1 - model.adversarial_loss_scale
            ) * recon_loss - model.adversarial_loss_scale * discriminator_loss

            update_encoder = True
            update_discriminator = True
            update_decoder = True

            # margins for training stability
            if (
                original_dis_cost.mean() < 0.68 - 0.4#self.equilibrium - self.margin
                or prior_dis_cost.mean() < 0.68 - 0.4#self.equilibrium - self.margin
            ):
                update_discriminator = False
    
            if (
                original_dis_cost.mean() > 0.68 + 0.4 #self.equilibrium + self.margin
                or prior_dis_cost.mean() > 0.68 + 0.4#self.equilibrium + self.margin
            ):
                update_decoder = False
    
            if not update_decoder and not update_discriminator:
                update_discriminator = True
                update_decoder = True

        else:

            F_loss = recon_loss_dict[model.recon_losses[m]]

            
            loss_1 = F_loss(
                    recons[m].reshape(xm.shape[0], -1),
                    xm.reshape(xm.shape[0], -1),
                    reduction="none",
                ).sum(dim=-1).mean(dim=0)*model.lik_scaling[m]

            details[f'loss_{m}'] = loss_1.item()

            encoder_loss = encoder_loss - loss_1#details[f'loss_{m}']
            decoder_loss = decoder_loss - loss_1#details[f'loss_{m}']


    details['encoder_loss'] = encoder_loss.item()
    details['decoder_loss'] = decoder_loss.item()
    details['discriminator_loss'] = discriminator_loss.item()
    # KLD to the prior
    mu, log_var = qz_xy.mean, 2*torch.log(qz_xy.stddev)
    # print(list(model.joint_encoder.parameters())[0].requires_grad)
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean(dim=0)
    details['kld_prior'] = kld.item()
    # Approximate the posterior
    if epoch >= warmup:
        reg, det = model.compute_kld(x)
        details['reg'] = reg
        update_details(details, det)
    else :
        details['reg']=0

    if epoch >= warmup:
        return (- details['reg'], details)

    else:
        encoder_loss = encoder_loss - beta_prior*kld
        return (
            encoder_loss,
            decoder_loss,
            discriminator_loss,
            update_encoder,
            update_decoder,
            update_discriminator,  
            details)



def m_telbo_nf(model,x,K=1, epoch=1, warmup=0, beta_prior=1):
    if epoch >= warmup:
        model.joint_encoder.requires_grad_(not model.fix_jencoder) #fix the joint encoder
        for vae in model.vaes:
            vae.decoder.requires_grad_(not model.fix_decoders) #fix the decoders
    qz_xy, recons, z_xy = model.forward(x)
    # mu, std = model.joint_encoder.forward(x)
    loss, details = 0, {}
    for m, xm in enumerate(x):
        assert recons[m].shape == xm.shape , f'Sizes are different : {recons[m].shape,xm.shape}'

        loss = loss - F.mse_loss(
                recons[m].reshape(xm.shape[0], -1),
                xm.reshape(xm.shape[0], -1),
                reduction="none",
            ).sum(dim=-1).sum()*model.lik_scaling[m]
    details['loss'] = loss.item()
    # KLD to the prior
    mu, log_var = qz_xy.mean, 2*torch.log(qz_xy.stddev)
    # print(list(model.joint_encoder.parameters())[0].requires_grad)
    details['kld_prior'] = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).sum()
    # Approximate the posterior
    if epoch >= warmup:
        # Add the unimodal elbos
        for m,vae in enumerate(model.vaes):
            o = model.vaes[m].forward(x[m])
            details[f'recon_loss_{m}'] = o.recon_loss * x[m].shape[0]
            details[f'kld_{m}'] = o.kld *x[m].shape[0]
            loss -= (details[f'recon_loss_{m}'] + model.beta_kl*details[f'kld_{m}'])*model.lik_scaling[m]

    return loss - beta_prior * details['kld_prior'] , details

def m_multi_elbos(model, x, K=1, beta=0):
    """ Generalized multimodal Elbo loss introduced in (Sutter 2021).
    It consists in sum of modified ELBOS that minimizes a sum of KL diver'gence"""

    if not hasattr(model, 'joint_encoder'):
        raise TypeError('The model must have a joint encoder for this loss.')
    qz_xy, pxy_z, z_xy = model.forward_joint(x,K=1)
    qz_xs, px_zs, z_xs = model.forward(x, K=1)
    loss = 0
    n_modal = len(pxy_z)
    if n_modal != 2:
        print("This loss is not normalized correctly for more than 2 modalities")
    for m in range(n_modal):
        loss = loss + pxy_z[m].log_prob(x[m]).squeeze().mean(0).sum()
        for r in range(n_modal):
            loss += px_zs[r][m].log_prob(x[m]).squeeze().mean(0).sum()

        loss -= kl_divergence(qz_xs[m],model.pz(*model.pz_params)).mean(0).sum()
    loss -= kl_divergence(qz_xy, model.pz(*model.pz_params)).mean(0).sum()

    return loss/3, {}


def m_svae(model, x, K=1, beta = 0):
    """ Loss implemented at the same time in the SVAE (in defense of product of experts) and VAEVAE article"""

    if not hasattr(model, 'joint_encoder'):
        raise TypeError('The model must have a joint encoder for this loss.')
    qz_xy, pxy_z, z_xy = model.forward_joint(x,K=1)
    qz_xs, px_zs, z_xs = model.forward(x, K=1)
    loss = 0

    loss, reg = 0, 0
    n_modal = len(pxy_z)
    for m in range(n_modal):
        # unimodal elbos
        loss += px_zs[m][m].log_prob(x[m]).mean()
        reg += kl_divergence(qz_xs[m], model.pz(*model.pz_params)).mean(0).sum()
        # joint reconstruction
        loss += pxy_z[m].log_prob(x[m]).mean()
        reg += kl_divergence(qz_xy, qz_xs[m]).mean(0).sum()

    return 1/2*(loss - beta*reg) , dict(loss=loss, reg=reg)


def m_telbo(model, x, K=1, beta=0, epoch=1, warmup=0, beta_prior = 1):
    """ Loss implemented in "Generative models of visually grounded imagination"
    We optimize simultaneously the unimodal elbos and the multimodal ones."""

    if not hasattr(model, 'joint_encoder'):
        raise TypeError('The model must have a joint encoder for this loss.')

    qz_xy, pxy_z, z_xy = model.forward_joint(x,K=1)
    qz_xs, px_zs, z_xs = model.forward(x, K=1)
    details = {'mloss' : 0}
    for m in range(len(pxy_z)):
        # unimodal elbos : fix parameter theta for this computation
        model.vaes[m].enc.requires_grad_(False)
        details[f'loss_{m}'] = px_zs[m][m].log_prob(x[m]).squeeze().mean(0).sum()
        details[f'loss_{m}'] -= beta_prior*kl_divergence(qz_xs[m], model.pz(*model.pz_params)).mean(0).sum()

        # joint reconstruction
        model.vaes[m].enc.requires_grad_(True)
        details['mloss'] += pxy_z[m].log_prob(x[m]).squeeze().mean(0).sum()
    # Add multimodal kl
    details['reg'] = beta_prior*kl_divergence(qz_xy, model.pz(*model.pz_params)).mean(0).sum()

    loss = details['mloss'] - details['reg'] + beta*(details['loss_0'] + details['loss_1'])
    return loss, details



def m_iwae(model, x, K=1, beta=0):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw = [_m_iwae(model, _x, K) for _x in x_split]
    lw = torch.cat(lw, 1)  # concat on batch
    details = {}
    return log_mean_exp(lw).sum(), details


def _m_iwae_looser(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.stack(lws)  # (n_modality * n_samples) x batch_size, batch_size


def m_iwae_looser(model, x, K=1, beta = 0):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw = [_m_iwae_looser(model, _x, K) for _x in x_split]
    lw = torch.cat(lw, 2)  # concat on batch
    return log_mean_exp(lw, dim=1).mean(0).sum()


def _m_dreg(model, x, K=1, beta=0):
    """DERG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
    qz_xs, px_zs, zss, qz_x_params = model(x, K)
    # Stop grad --> detach the parameters so that the gradient doesn't retropropagate on this
    qz_xs_ = [model.qz_x(*[p.detach() for p in qz_x_params[i]]) for i in range(len(model.vaes))]
    lws = []
    for r, vae in enumerate(model.vaes):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.lik_scaling[d]).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        # print(lpx_z[0].sum(), lpx_z[1].sum())
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.cat(lws), torch.cat(zss)


def m_dreg(model, x, K=1, beta=0, warmup=0, epoch=1, beta_prior=1):
    """Computes dreg estimate for log p_\theta(x) for multi-modal vae """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw, zss = zip(*[_m_dreg(model, _x, K) for _x in x_split])
    lw = torch.cat(lw, 1)  # concat on batch
    zss = torch.cat(zss, 1)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 0, keepdim=True)).exp()
        if zss.requires_grad:
            zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad) # Multiply the gradient by (w_i / sum_j w_j) to get the Dreg gradient
    details = {}
    return (grad_wt * lw).sum(), details


def _m_dreg_looser(model, x, K=1, beta=0):
    """DERG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    qz_xs, px_zs, zss, qz_x_params = model(x, K)
    qz_xs_ = [model.qz_x(*[p.detach() for p in qz_x_params[i]]) for i in range(len(model.vaes))]
    lws = []
    for r, vae in enumerate(model.vaes):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.lik_scaling[d]).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.stack(lws), torch.stack(zss)


def m_dreg_looser(model, x,K,epoch,warmup, beta_prior):
    """Computes dreg estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw, zss = zip(*[_m_dreg_looser(model, _x, K) for _x in x_split])
    lw = torch.cat(lw, 2)  # concat on batch
    zss = torch.cat(zss, 2)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
        if zss.requires_grad:
            zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).mean(0).sum(), {}


def m_elbo_nf_(model, x, K, epoch, warmup, beta_prior):

    ln_qz_xs, zs, recons = model.forward(x)
    # Compute reconstruction terms
    lpxy_z = [ [-1/2*torch.sum((x[i] - recon)**2)
               for i,recon in enumerate(recon_row)] for recon_row in recons ]
    lpxy_z = ((lpxy_z[0][0] + lpxy_z[1][0])*model.lik_scaling[0] +(lpxy_z[0][1]  + lpxy_z[1][1])*model.lik_scaling[1])/2

    # print(lpxy_z)
    # Compute KL divergence

    sum_ln_qz_xs = torch.sum(torch.stack([torch.logsumexp(torch.stack(row), dim=0).sum() for row in ln_qz_xs]))/2

    # We assume the prior to be a standard gaussian
    ln_p_z = torch.sum(torch.Tensor([-1/2*torch.sum(z**2) for z in zs]))

    kld = sum_ln_qz_xs - ln_p_z
    # print(kld)
    print(lpxy_z -kld)
    return lpxy_z - kld, {}


def m_elbo_nf(model, x, K, epoch, warmup, beta_prior):
    """ ELBO computation developing the KLD """

    ln_qz_xs, zs, recons = model.forward(x)
    elbo = 0
    for e, row in enumerate(ln_qz_xs):

        # Compute the KL
        log_prob_z = -1/2*torch.sum(zs[e]**2)
        kld = row[e].sum() - log_prob_z
        elbo -= kld/len(model.vaes)
        # Compute reconstruction errors
        for d, recon in enumerate(recons[e]):
            # We assume gaussian distribution for decoder
            elbo += (-1/2*torch.sum((recon-x[d])**2)) /len(model.vaes) * model.lik_scaling[d]

    return elbo / len(x) , {}

def m_self_built(model, x, K, epoch, warmup, beta_prior):
    elbo = model.forward(x)['elbo']
    return elbo, {}