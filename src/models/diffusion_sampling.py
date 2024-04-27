# Code taken from: https://github.com/crowsonkb/v-diffusion-pytorch

import torch
from tqdm.auto import trange

import src.models.diffusion_utils as utils
import pdb
from torchvision.utils import save_image
import torch.nn.functional as F


def plot_tensor(x):
    import matplotlib.pyplot as plt
    if x.ndim == 3:
        fig, ax = plt.subplots(1, 1)
        axs = [ax]
        x = x.unsqueeze(0)
    if x.ndim == 4:
        num_imgs = x.shape[0]
        import math
        num_rows = math.ceil(math.sqrt(num_imgs))
        fig, axs = plt.subplots(num_rows, num_rows)
        axs = axs.flatten()
    for ax, img in zip(axs, x):
        img -= img.min()
        img /= img.max()
        ax.imshow(img.detach().cpu().numpy().transpose(1, 2, 0))
        ax.axis('off')
    fig.tight_layout()
    fig.savefig("sample.png")
    plt.show()


# @torch.no_grad()
def sample_guidance(model, x, classifier, label, steps, eta, extra_args, classifier_guidance_scale=1e3, callback=None):
    """Draws samples from a model given starting noise."""

    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = utils.t_to_alpha_sigma(steps)

    # The sampling loop
    for i in trange(len(steps), disable=None):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * steps[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]

        # # if shape of x is not 224x224, resize it to 224x224
        # if pred.shape[-1] != 224:
        #     pred_reshaped = torch.nn.functional.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=False)

        if i % 100 == 0 and i != 0:
            save_image(pred, f"test/pred_test_{i}.png")

        # Calculate the classifier output and the cross-entropy loss
        with torch.cuda.amp.autocast():
            # pred_reshaped = (pred_reshaped - pred_reshaped.min()) / (pred_reshaped.max() - pred_reshaped.min())
            logits = classifier(pred)
            loss = F.cross_entropy(logits, label)
            loss = loss * classifier_guidance_scale

        # Compute gradients to guide the sampling
        grads = torch.autograd.grad(loss, x)[0]

        # if i % 100 == 0 and i!=0:
        #     pdb.set_trace()
        if steps[i] < 1:
            v = v.detach() - grads
        else:
            v = v.detach()

        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # Call the callback
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'v': v, 'pred': pred})

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < len(steps) - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred


# DDPM/DDIM sampling
@torch.no_grad()
def sample(model, x, steps, eta, extra_args, callback=None):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = utils.t_to_alpha_sigma(steps)

    # The sampling loop
    for i in trange(len(steps), disable=None):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * steps[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # Call the callback
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'v': v, 'pred': pred})

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < len(steps) - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred


@torch.no_grad()
def cond_sample(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    steps: torch.Tensor,
    eta: float,
    extra_args: dict=dict(),
    classifier: torch.nn.Module,
    label: torch.Tensor,
    num_backward_steps: int=10,
    backward_step_size: float=1e-1,
    backward_guidance_scale: float=1e-1,
    forward_guidance_scale: float=1e-1,
    verbose: bool=True
) -> torch.Tensor:
    
    """Draws guided samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = utils.t_to_alpha_sigma(steps)

    # The sampling loop
    for i in trange(len(steps), disable=not verbose):

        # check for nans
        if torch.isnan(x).any():
            print(i)
            raise ValueError("x contains NaNs")

        # Get the model output
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            with torch.cuda.amp.autocast():
                v = model(x, ts * steps[i], **extra_args)

            pred = x * alphas[i] - v * sigmas[i]

            if steps[i] < 1:
                _pred = pred.clone().detach().requires_grad_(True)
                logits = classifier(_pred)
                loss = torch.nn.functional.cross_entropy(logits, label)
                loss.backward()
                cond_grad = forward_guidance_scale * _pred.grad
                v = (v - cond_grad * (sigmas[i] / alphas[i])).detach()
            else:
                v = v.detach()

        if num_backward_steps > 0:        
            pred = x * alphas[i] - v * sigmas[i]

            with torch.enable_grad():
                _pred = pred.clone().detach().requires_grad_(True)

                # backward universal guidance
                delta = torch.zeros_like(_pred)
                for _ in range(num_backward_steps):
                    delta = delta.detach().clone().requires_grad_(True)

                    logits = classifier(_pred + delta)
                    loss = torch.nn.functional.cross_entropy(logits, label)
                    loss.backward()

                    delta = delta - backward_step_size * delta.grad

                v = (v - delta * torch.sqrt(alphas[i] / (1 - alphas[i])))

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < len(steps) - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred


@torch.no_grad()
def reverse_sample(model, x, steps, extra_args, callback=None):
    """Finds a starting latent that would produce the given image with DDIM
    (eta=0) sampling."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = utils.t_to_alpha_sigma(steps)

    # The sampling loop
    for i in trange(len(steps) - 1, disable=None):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * steps[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # Call the callback
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'v': v, 'pred': pred})

        # Recombine the predicted noise and predicted denoised image in the
        # correct proportions for the next step
        x = pred * alphas[i + 1] + eps * sigmas[i + 1]

    return x


# PNDM sampling (see https://openreview.net/pdf?id=PlKWVd2yBkY)

def make_eps_model_fn(model):
    def eps_model_fn(x, t, **extra_args):
        alphas, sigmas = utils.t_to_alpha_sigma(t)
        v = model(x, t, **extra_args)
        eps = x * utils.append_dims(sigmas, x.ndim) + v * utils.append_dims(alphas, x.ndim)
        return eps
    return eps_model_fn


def make_autocast_model_fn(model, enabled=True):
    def autocast_model_fn(*args, **kwargs):
        with torch.cuda.amp.autocast(enabled):
            return model(*args, **kwargs).float()
    return autocast_model_fn


def transfer(x, eps, t_1, t_2):
    alphas, sigmas = utils.t_to_alpha_sigma(t_1)
    next_alphas, next_sigmas = utils.t_to_alpha_sigma(t_2)
    pred = (x - eps * utils.append_dims(sigmas, x.ndim)) / utils.append_dims(alphas, x.ndim)
    x = pred * utils.append_dims(next_alphas, x.ndim) + eps * utils.append_dims(next_sigmas, x.ndim)
    return x, pred


def prk_step(model, x, t_1, t_2, extra_args):
    eps_model_fn = make_eps_model_fn(model)
    t_mid = (t_2 + t_1) / 2
    eps_1 = eps_model_fn(x, t_1, **extra_args)
    x_1, _ = transfer(x, eps_1, t_1, t_mid)
    eps_2 = eps_model_fn(x_1, t_mid, **extra_args)
    x_2, _ = transfer(x, eps_2, t_1, t_mid)
    eps_3 = eps_model_fn(x_2, t_mid, **extra_args)
    x_3, _ = transfer(x, eps_3, t_1, t_2)
    eps_4 = eps_model_fn(x_3, t_2, **extra_args)
    eps_prime = (eps_1 + 2 * eps_2 + 2 * eps_3 + eps_4) / 6
    x_new, pred = transfer(x, eps_prime, t_1, t_2)
    return x_new, eps_prime, pred


def plms_step(model, x, old_eps, t_1, t_2, extra_args):
    eps_model_fn = make_eps_model_fn(model)
    eps = eps_model_fn(x, t_1, **extra_args)
    eps_prime = (55 * eps - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24
    x_new, _ = transfer(x, eps_prime, t_1, t_2)
    _, pred = transfer(x, eps, t_1, t_2)
    return x_new, eps, pred


@torch.no_grad()
def prk_sample(model, x, steps, extra_args, is_reverse=False, callback=None):
    """Draws samples from a model given starting noise using fourth-order
    Pseudo Runge-Kutta."""
    ts = x.new_ones([x.shape[0]])
    model_fn = make_autocast_model_fn(model)
    if not is_reverse:
        steps = torch.cat([steps, steps.new_zeros([1])])
    for i in trange(len(steps) - 1, disable=None):
        x, _, pred = prk_step(model_fn, x, steps[i] * ts, steps[i + 1] * ts, extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'pred': pred})
    return x


@torch.no_grad()
def plms_sample(model, x, steps, extra_args, is_reverse=False, callback=None):
    """Draws samples from a model given starting noise using fourth order
    Pseudo Linear Multistep."""
    ts = x.new_ones([x.shape[0]])
    model_fn = make_autocast_model_fn(model)
    if not is_reverse:
        steps = torch.cat([steps, steps.new_zeros([1])])
    old_eps = []
    for i in trange(len(steps) - 1, disable=None):
        if len(old_eps) < 3:
            x, eps, pred = prk_step(model_fn, x, steps[i] * ts, steps[i + 1] * ts, extra_args)
        else:
            x, eps, pred = plms_step(model_fn, x, old_eps, steps[i] * ts, steps[i + 1] * ts, extra_args)
            old_eps.pop(0)
        old_eps.append(eps)
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'pred': pred})
    return x


def pie_step(model, x, t_1, t_2, extra_args):
    eps_model_fn = make_eps_model_fn(model)
    eps_1 = eps_model_fn(x, t_1, **extra_args)
    x_1, _ = transfer(x, eps_1, t_1, t_2)
    eps_2 = eps_model_fn(x_1, t_2, **extra_args)
    eps_prime = (eps_1 + eps_2) / 2
    x_new, pred = transfer(x, eps_prime, t_1, t_2)
    return x_new, eps_prime, pred


def plms2_step(model, x, old_eps, t_1, t_2, extra_args):
    eps_model_fn = make_eps_model_fn(model)
    eps = eps_model_fn(x, t_1, **extra_args)
    eps_prime = (3 * eps - old_eps[-1]) / 2
    x_new, _ = transfer(x, eps_prime, t_1, t_2)
    _, pred = transfer(x, eps, t_1, t_2)
    return x_new, eps, pred


@torch.no_grad()
def pie_sample(model, x, steps, extra_args, is_reverse=False, callback=None):
    """Draws samples from a model given starting noise using second-order
    Pseudo Improved Euler."""
    ts = x.new_ones([x.shape[0]])
    model_fn = make_autocast_model_fn(model)
    if not is_reverse:
        steps = torch.cat([steps, steps.new_zeros([1])])
    for i in trange(len(steps) - 1, disable=None):
        x, _, pred = pie_step(model_fn, x, steps[i] * ts, steps[i + 1] * ts, extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'pred': pred})
    return x


@torch.no_grad()
def plms2_sample(model, x, steps, extra_args, is_reverse=False, callback=None):
    """Draws samples from a model given starting noise using second order
    Pseudo Linear Multistep."""
    ts = x.new_ones([x.shape[0]])
    model_fn = make_autocast_model_fn(model)
    if not is_reverse:
        steps = torch.cat([steps, steps.new_zeros([1])])
    old_eps = []
    for i in trange(len(steps) - 1, disable=None):
        if len(old_eps) < 1:
            x, eps, pred = pie_step(model_fn, x, steps[i] * ts, steps[i + 1] * ts, extra_args)
        else:
            x, eps, pred = plms2_step(model_fn, x, old_eps, steps[i] * ts, steps[i + 1] * ts, extra_args)
            old_eps.pop(0)
        old_eps.append(eps)
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'pred': pred})
    return x


def iplms_step(model, x, old_eps, t_1, t_2, extra_args):
    eps_model_fn = make_eps_model_fn(model)
    eps = eps_model_fn(x, t_1, **extra_args)
    if len(old_eps) == 0:
        eps_prime = eps
    elif len(old_eps) == 1:
        eps_prime = (3/2 * eps - 1/2 * old_eps[-1])
    elif len(old_eps) == 2:
        eps_prime = (23/12 * eps - 16/12 * old_eps[-1] + 5/12 * old_eps[-2])
    else:
        eps_prime = (55/24 * eps - 59/24 * old_eps[-1] + 37/24 * old_eps[-2] - 9/24 * old_eps[-3])
    x_new, _ = transfer(x, eps_prime, t_1, t_2)
    _, pred = transfer(x, eps, t_1, t_2)
    return x_new, eps, pred


@torch.no_grad()
def iplms_sample(model, x, steps, extra_args, is_reverse=False, callback=None):
    """Draws samples from a model given starting noise using fourth order
    Improved Pseudo Linear Multistep."""
    ts = x.new_ones([x.shape[0]])
    model_fn = make_autocast_model_fn(model)
    if not is_reverse:
        steps = torch.cat([steps, steps.new_zeros([1])])
    old_eps = []
    for i in trange(len(steps) - 1, disable=None):
        x, eps, pred = iplms_step(model_fn, x, old_eps, steps[i] * ts, steps[i + 1] * ts, extra_args)
        if len(old_eps) >= 3:
            old_eps.pop(0)
        old_eps.append(eps)
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'pred': pred})
    return x