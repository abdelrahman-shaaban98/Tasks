import torch
import torch.nn as nn
from torchvision.models import vgg19_bn, VGG19_BN_Weights

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 3
        latent_dim = 128

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=3,
                                      kernel_size= 3, padding=1),
                            nn.Tanh())

        self.feature_network = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        # Freeze the pretrained feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False

        self.feature_network.eval()

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)    
        result = self.decoder(result)
        result = self.final_layer(result)       
        return result

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        recons_features = self.extract_features(recons)
        x_features = self.extract_features(x)
        return  [z, recons, x_features, recons_features, mu, log_var]

    def extract_features(self, x):
        feature_layers = ['14', '24', '34', '43']
        features = []
        result = x
        
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if(key in feature_layers):
                features.append(result)

        return features