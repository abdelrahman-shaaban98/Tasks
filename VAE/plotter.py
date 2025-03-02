import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_loss(losses, num_epochs, averaging_iterations=100, label=''):
    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(losses)),
             (losses), label=f'Loss {label}')

    if len(losses) < 50:
        num_losses = len(losses) // 2
    else:
        num_losses = 50

    ax1.set_ylim([0, np.max(losses[num_losses:])])

    ax1.plot(np.convolve(losses,
                         np.ones(averaging_iterations,) / averaging_iterations,
                         mode='valid'),
             label=f'Running Average{label}')
    
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.legend()

    plt.tight_layout()
    
def plot_generated_images(data_loader, model, device, n_images=15):
    figsize=(20, 2.5)
    fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                             sharex=True, sharey=True, figsize=figsize)
    
    for batch_idx, (x, _) in enumerate(data_loader):
        x = x.to(device)
        
        with torch.no_grad():
            encoded, decoded_images, x_features, decoded_features, z_mean, z_log_var = model(x)[:n_images]

        orig_images = x[:n_images]
        break

    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            curr_img = img[i].detach().to('cpu')        
            curr_img = np.transpose(curr_img, (1, 2, 0))
            ax[i].imshow(curr_img)
            ax[i].axis('off')


def plot_images_sampled_from_vae(model, device, latent_size, num_images=10):
    with torch.no_grad():
        rand_features = torch.randn(num_images, latent_size).to(device)
        new_images = model.decode(rand_features)
        
        fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(10, 2.5))
        decoded_images = new_images[:num_images]

        for ax, img in zip(axes, decoded_images):
            curr_img = img.detach().to(torch.device('cpu'))        
            curr_img = np.transpose(curr_img, (1, 2, 0)).clip(0, 1)
            ax.imshow(curr_img)
            ax.axis('off')


def plot_embedding_hist(all_embeddings):
    fig, axes = plt.subplots(nrows=20, ncols=5, sharex=True, sharey=True, figsize=(10, 15))
    
    i = 0
    for row in range(len(axes)):
        for col in range(len(axes[0])):
            axes[row][col].hist(all_embeddings[:, i].numpy())
            i += 1


def plot_modified_faces(original, diff,
                        diff_coefficients=(0., 0.5, 1., 1.5, 2., 2.5, 3.),
                        decoding_fn=None,
                        device=None,
                        figsize=(8, 2.5)):

    fig, axes = plt.subplots(nrows=2, ncols=len(diff_coefficients), figsize=figsize)
    

    for i, alpha in enumerate(diff_coefficients):
        more = original + alpha * diff
        less = original - alpha * diff
        
        if decoding_fn is not None:
            with torch.no_grad():
                if device is not None:
                    more = more.to(device).unsqueeze(0)
                    less = less.to(device).unsqueeze(0)

                more = decoding_fn(more).to('cpu').squeeze(0)
                less = decoding_fn(less).to('cpu').squeeze(0)
        
        if not alpha:
            s = 'original'
        else:
            s = f'$\\alpha=${alpha}'
            
        axes[0][i].set_title(s)
        axes[0][i].imshow(more.permute(1, 2, 0))
        axes[1][i].imshow(less.permute(1, 2, 0))
        axes[1][i].axis('off')
        axes[0][i].axis('off')