import torch
from argparse import Namespace

ppo_config = Namespace(
     vf_coef  = 0.5,
     ent_coef  = 0.01,
     anneal_clipping  = True,
     clip_coef  = 0.1,
     norm_adv  = True, 
     update_epochs  = 4,
     gae_lambda = 0.95,
     gamma  = 0.99,
     num_steps = 128,
     max_grad_norm = 0.5,
     learning_rate = 2.5e-4,
     decay_steps = 50000,)

def compute_ppo_loss(
    logprobs, new_logprobs, values, new_values, advantages, returns, entropy, clip_coef=0.2, clip_vf_coef=None):
    logratio = new_logprobs - logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1.0) - logratio).mean()
        clipfrac = [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    new_values = new_values.view(-1)
    if clip_vf_coef is None:
        v_loss = 0.5 * ((new_values - returns) ** 2).mean()
    else:
        v_loss_unclipped = (new_values - returns) ** 2
        v_clipped = values + torch.clamp(
            new_values - values, -clip_vf_coef, clip_vf_coef)
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    entropy_loss = entropy.mean()
    return pg_loss, entropy_loss, v_loss, approx_kl, old_approx_kl, clipfrac
