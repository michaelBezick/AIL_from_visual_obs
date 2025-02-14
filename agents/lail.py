import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch import distributions as torchd
from torch import autograd
from torch.nn.utils import spectral_norm 
from torchvision import transforms

from utils_folder import utils
from utils_folder.utils_dreamer import Bernoulli
from utils_folder.resnet import BasicBlock, ResNet84

from transfer_learning.LucasKanadeOptFlow import optical_flow
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import raft_small
import torchvision.transforms as T
import torchvision.transforms.functional as FT
from torchvision.utils import flow_to_image

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=8, num_channels=self.in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

def inRange( cordinates, limits):
	x,y = cordinates
	X_Limit, Y_Limit = limits
	return 0 <= x and x < X_Limit and 0 <= y and y < Y_Limit

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class NoAug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
class Identity(nn.Module):
    def __init__(self, input_placeholder=None):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class PretrainedEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, model_name, device):
        super().__init__()
        # a wrapper over a non-RL encoder model
        self.device = device
        assert len(obs_shape) == 3
        self.n_input_channel = obs_shape[0]
        assert self.n_input_channel % 3 == 0
        self.n_images = self.n_input_channel // 3
        self.model = self.init_model(model_name)
        self.model.fc = Identity()
        self.repr_dim = self.model.get_feature_size()

        self.normalize_op = transforms.Normalize((0.485, 0.456, 0.406),
                                                 (0.229, 0.224, 0.225))
        self.channel_mismatch = True

        self.trunk = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
                        nn.LayerNorm(feature_dim), nn.Tanh())
        
        self.apply(utils.weight_init)

    def init_model(self, model_name):
        # model name is e.g. resnet6_32channel
        n_layer_string, n_channel_string = model_name.split('_')

        layer_string_to_layer_list = {
            'resnet6': [0, 0, 0, 0],
            'resnet10': [1, 1, 1, 1],
            'resnet18': [2, 2, 2, 2],
        }

        channel_string_to_n_channel = {
            '32channel': 32,
            '64channel': 64,
        }

        layer_list = layer_string_to_layer_list[n_layer_string]
        start_num_channel = channel_string_to_n_channel[n_channel_string]
        return ResNet84(BasicBlock, layer_list, start_num_channel=start_num_channel).to(self.device)

    def expand_first_layer(self):
        # convolutional channel expansion to deal with input mismatch
        multiplier = self.n_images
        self.model.conv1.weight.data = self.model.conv1.weight.data.repeat(1,multiplier,1,1) / multiplier
        means = (0.485, 0.456, 0.406) * multiplier
        stds = (0.229, 0.224, 0.225) * multiplier
        self.normalize_op = transforms.Normalize(means, stds)
        self.channel_mismatch = False

    def freeze_bn(self):
        # freeze batch norm layers (VRL3 ablation shows modifying how
        # batch norm is trained does not affect performance)
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    def get_parameters_that_require_grad(self):
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                params.append(param)
        return params

    def transform_obs_tensor_batch(self, obs):
        # transform obs batch before put into the pretrained resnet
        new_obs = self.normalize_op(obs.float()/255)
        return new_obs

    def _forward_impl(self, x):
        x = self.model.get_features(x)
        return x

    def forward(self, obs):
        o = self.transform_obs_tensor_batch(obs)
        h = self._forward_impl(o)
        z = self.trunk(h)
        return z

class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3

        self.repr_dim = 32 * 35 * 35
        self.feature_dim = (32,35,35)

        self.additional_dim_optical_flow = 2

        self.initial_conv = nn.Conv2d(obs_shape[0] + self.additional_dim_optical_flow, 32, 3, stride=2)
        self.relu = nn.ReLU()
        self.convnet = nn.Sequential(nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        
        self.trunk = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
                                nn.LayerNorm(feature_dim), nn.Tanh())

        self.apply(utils.weight_init)

        self.counter = 0
        self.optical_flow_model = raft_small(pretrained=True, progress=False).cuda()
        self.optical_flow_model = self.optical_flow_model.eval()
        self.attention = AttnBlock(32)
        # self.transform = T.Compose([T.Resize(size=(160,160))])

    # def _optical_flow(self, obs):
    #
    #     #obs = torch.squeeze(obs)
    #
    #     num_frames = obs.size()[1] // 3
    #     num_pairs = num_frames-1
    #
    #     og_obs = obs
    #
    #     return_tensor = torch.zeros(obs.size()[0], num_pairs * 2, obs.size()[2], obs.size()[3])
    #
    #     for batch_dim in range(og_obs.size()[0]):
    #
    #         obs = og_obs[batch_dim, :, :, :]
    #
    #         pairs = []
    #
    #         for i in range(num_pairs):
    #             pairs.append((obs[i * 3 : i * 3 + 3, :, :], obs[(i + 1) * 3 :(i + 1) * 3 + 3, :, :]))
    #
    #         optical_flows = None
    #
    #         for i, pair in enumerate(pairs):
    #             img1, img2 = pair
    #
    #             img1 = torch.squeeze(img1)
    #             img2 = torch.squeeze(img2)
    #
    #             img1 = torch.permute(img1, (1, 2, 0))
    #             img2 = torch.permute(img2, (1, 2, 0))
    #
    #             img1 = img1.detach().cpu().numpy()
    #             img2= img2.detach().cpu().numpy()
    #
    #             img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #             img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #
    #             U,V = optical_flow(img1, img2, 8, 0.005)
    #
    #             U = torch.tensor(U).unsqueeze(0)
    #             V = torch.tensor(V).unsqueeze(0)
    #
    #             if i == 0:
    #                 optical_flows = torch.cat([U, V], dim=0)
    #             else:
    #                 optical_flows = torch.cat([optical_flows, U], dim=0)
    #                 optical_flows = torch.cat([optical_flows, V], dim=0)
    #
    #         return_tensor[batch_dim, :, :, :] = optical_flows
    #
    #     return return_tensor
                



    def min_max_norm(self, x):
        if torch.min(x) != torch.max(x):
            return (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        else:
            return x

    # def save_optical_flow(self, old_frame, new_frame, U, V, output_file):
    #
    #     img2 = new_frame
    #     img1 = old_frame
    #
    #     displacement = np.ones_like(img2)
    #     displacement.fill(255.)             #Fill the displacement plot with White background
    #     line_color =  (0, 0, 0)
    #     # draw the displacement vectors
    #     for i in range(img2.shape[0]):
    #         for j in range(img2.shape[1]):
    #
    #             start_pixel = (i,j)
    #             end_pixel = ( int(i+U[i][j]), int(j+V[i][j]) )
    #
    #             #check if there is displacement for the corner and endpoint is in range
    #             if U[i][j] and V[i][j] and inRange( end_pixel, img1.shape ):     
    #                 displacement = cv2.arrowedLine( displacement, start_pixel, end_pixel, line_color, thickness =2)
    #
    #     figure, axes = plt.subplots(1,3)
    #     axes[0].imshow(old_frame, cmap = "gray")
    #     axes[0].set_title("first image")
    #     axes[1].imshow(new_frame, cmap = "gray")
    #     axes[1].set_title("second image")
    #     axes[2].imshow(displacement, cmap = "gray")
    #     axes[2].set_title("displacements")
    #     figure.tight_layout()
    #     plt.savefig(output_file, bbox_inches = "tight", dpi = 200)

    def optical_flow(self, x):
        x = self.min_max_norm(x) * 2 - 1
        x = FT.resize(x, size=(88,88))
        flow = self.optical_flow_model(x[:, -3:, :, :], x[:, -6:-3, :, :])
        flow = flow[-1]
        flow = FT.resize(flow, size=(84,84))
        
        assert not torch.isnan(flow).any()

        return flow

    def plot(self, imgs, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                img = FT.to_pil_image(img.to("cpu"))
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.savefig("predicted_flows.jpg")


    def forward(self, obs):
        #I believe that observation window is always 3 frames
        #For walker_walk task, it is 9x84x84, which is 3 RGB images.
        obs = obs / 255.0 - 0.5

        with torch.no_grad():
            flow = self.optical_flow(obs)

        flow = self.min_max_norm(flow)

        flow = flow - 1


        """new plotting"""
        #flow = flow_to_image(flow)

        #new_obs = obs[:, -3:, :, :]
        #new_obs = self.min_max_norm(new_obs)

        #img = [img for img in new_obs]

        #grid = [[img, flow_img] for (img, flow_img) in zip(img, flow)]

        #self.plot(grid)
        #exit()

        """end"""


        #The observation shape input to encoder is [9, 84, 84]
        obs = torch.cat([obs, flow], dim=1)
        h = self.initial_conv(obs)
        h = self.attention(h)
        h = self.relu(h)

        h = self.convnet(h)
        h = h.view(h.shape[0], -1)

        z = self.trunk(h)

        return z
        
class Discriminator(nn.Module):
    def __init__(self, input_net_dim, hidden_dim, spectral_norm_bool=False, dist=None):
        super().__init__()
                
        self.dist = dist
        self._shape = (1,)

        if spectral_norm_bool:
            self.net = nn.Sequential(spectral_norm(nn.Linear(input_net_dim, hidden_dim)),
                                    nn.ReLU(inplace=True),
                                    spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                                    nn.ReLU(inplace=True),
                                    spectral_norm(nn.Linear(hidden_dim, 1)))  

        else:
            self.net = nn.Sequential(nn.Linear(input_net_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, 1))  
        
        self.apply(utils.weight_init)

    def forward(self, transition):
        d = self.net(transition)

        if self.dist == 'binary':
            return Bernoulli(torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=d), len(self._shape)))
        else:
            return d 

class Actor(nn.Module):
    def __init__(self, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

class Critic(nn.Module):
    def __init__(self, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2

class LailAgent:
    def __init__(self, 
                 obs_shape, 
                 action_shape, 
                 device, 
                 lr, 
                 feature_dim,
                 hidden_dim, 
                 critic_target_tau, 
                 num_expl_steps,
                 update_every_steps, 
                 stddev_schedule, 
                 stddev_clip, 
                 use_tb, 
                 reward_d_coef, 
                 discriminator_lr, 
                 spectral_norm_bool, 
                 pretrained_encoder_path, 
                 encoder_lr_scale, 
                 pretrained_encoder=False, 
                 pretrained_encoder_model_name = 'resnet6_32channel', 
                 GAN_loss='bce', 
                 from_dem=False, 
                 add_aug=True, 
                 RL_plus_IL = False):
        
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.GAN_loss = GAN_loss
        self.from_dem = from_dem
        self.RL_plus_IL = RL_plus_IL
        
        if pretrained_encoder:
            self.encoder = PretrainedEncoder(obs_shape, feature_dim, pretrained_encoder_model_name, device).to(device)
            self.load_pretrained_encoder(pretrained_encoder_path)
            self.encoder.expand_first_layer()
            print("Convolutional channel expansion finished: now can take in %d images as input." % self.encoder.n_images)
            encoder_lr = lr * encoder_lr_scale
            print("Using pretrained encoder")

        else:
            self.encoder = Encoder(obs_shape, feature_dim).to(device)
            encoder_lr = lr 
            print("Using new encoder")

        self.actor = Actor(action_shape, feature_dim, hidden_dim).to(device)
        self.critic = Critic(action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # added model
        if from_dem:
            if self.GAN_loss == 'least-square':
                self.discriminator = Discriminator(feature_dim+action_shape[0], hidden_dim, spectral_norm_bool).to(device)
                self.reward_d_coef = reward_d_coef

            elif self.GAN_loss == 'bce':
                self.discriminator = Discriminator(feature_dim+action_shape[0], hidden_dim, spectral_norm_bool, dist='binary').to(device)
            else:
                NotImplementedError

        else:
            if self.GAN_loss == 'least-square':
                self.discriminator = Discriminator(2*feature_dim, hidden_dim, spectral_norm_bool).to(device)
                self.reward_d_coef = reward_d_coef

            elif self.GAN_loss == 'bce':
                self.discriminator = Discriminator(2*feature_dim, hidden_dim, spectral_norm_bool, dist='binary').to(device)
            else:
                NotImplementedError

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr)

        # data augmentation
        if add_aug:
            self.aug = RandomShiftsAug(pad=4)
        else:
            self.aug = NoAug()

        self.train()
        self.critic_target.train()

    def load_pretrained_encoder(self, model_path, verbose=True):
        if verbose:
            print("Trying to load pretrained model from:", model_path)

        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        state_dict = checkpoint['state_dict']

        pretrained_dict = {}
        # remove `module.` if model was pretrained with distributed mode
        for k, v in state_dict.items():
            if 'module.' in k:
                name = k[7:]
            else:
                name = k
            pretrained_dict[name] = v

        self.encoder.model.load_state_dict(pretrained_dict, strict=False)

        if verbose:
            print("Pretrained model loaded!")

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.discriminator.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics
    
    def compute_reward(self, obs_a, next_a, reward_a):
        metrics = dict()

        # augment
        if self.from_dem:
            obs_a = self.aug(obs_a.float())
        else:
            obs_a = self.aug(obs_a.float())
            next_a = self.aug(next_a.float())
        
        # encode
        with torch.no_grad():
            if self.from_dem:
                obs_a = self.encoder(obs_a)
            else:
                obs_a = self.encoder(obs_a)
                next_a = self.encoder(next_a)
        
            self.discriminator.eval()
            transition_a = torch.cat([obs_a, next_a], dim = -1)

            d = self.discriminator(transition_a)

            if self.GAN_loss == 'least-square':
                reward_d = self.reward_d_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)

            elif self.GAN_loss == 'bce':
                reward_d = d.mode()
            
            if self.RL_plus_IL:
                reward = reward_d + reward_a

            else:
                reward = reward_d

            if self.use_tb:
                metrics['reward_d'] = reward_d.mean().item()
    
            self.discriminator.train()
            
        return reward, metrics
    
    def compute_discriminator_grad_penalty_LS(self, obs_e, next_e, lambda_=10):
        
        expert_data = torch.cat([obs_e, next_e], dim=-1)
        expert_data.requires_grad = True
        
        d = self.discriminator(expert_data)

        ones = torch.ones(d.size(), device=self.device)
        grad = autograd.grad(outputs=d, inputs=expert_data, grad_outputs=ones, create_graph=True,
                             retain_graph=True, only_inputs=True)[0]
        
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def compute_discriminator_grad_penalty_bce(self, obs_a, next_a, obs_e, next_e, lambda_=10):

        agent_feat = torch.cat([obs_a, next_a], dim=-1)
        alpha = torch.rand(agent_feat.shape[:1]).unsqueeze(-1).to(self.device)
        expert_data = torch.cat([obs_e, next_e], dim=-1)
        disc_penalty_input = alpha*agent_feat + (1-alpha)*expert_data

        disc_penalty_input.requires_grad = True

        d = self.discriminator(disc_penalty_input).mode()

        ones = torch.ones(d.size(), device=self.device)
        grad = autograd.grad(outputs=d, inputs=disc_penalty_input, grad_outputs=ones, create_graph=True,
                             retain_graph=True, only_inputs=True)[0]
        
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
        
    def update_discriminator(self, obs_a, next_a, obs_e, next_e):
        metrics = dict()

        transition_a = torch.cat([obs_a, next_a], dim=-1)
        transition_e = torch.cat([obs_e, next_e], dim=-1)
        
        agent_d = self.discriminator(transition_a)
        expert_d = self.discriminator(transition_e)

        if self.GAN_loss == 'least-square':
            expert_labels = 1.0
            agent_labels = -1.0

            expert_loss = F.mse_loss(expert_d, expert_labels*torch.ones(expert_d.size(), device=self.device))
            agent_loss = F.mse_loss(agent_d, agent_labels*torch.ones(agent_d.size(), device=self.device))
            grad_pen_loss = self.compute_discriminator_grad_penalty_LS(obs_e.detach(), next_e.detach())
            loss = 0.5*(expert_loss + agent_loss) + grad_pen_loss
        
        elif self.GAN_loss == 'bce':
            expert_loss = (expert_d.log_prob(torch.ones_like(expert_d.mode()).to(self.device))).mean()
            agent_loss = (agent_d.log_prob(torch.zeros_like(agent_d.mode()).to(self.device))).mean()
            grad_pen_loss = self.compute_discriminator_grad_penalty_bce(obs_a.detach(), next_a.detach(), obs_e.detach(), next_e.detach())
            loss = -(expert_loss+agent_loss) + grad_pen_loss

        self.discriminator_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_opt.step()
        
        if self.use_tb:
            metrics['discriminator_expert_loss'] = expert_loss.item()
            metrics['discriminator_agent_loss'] = agent_loss.item()
            metrics['discriminator_loss'] = loss.item()
            metrics['discriminator_grad_pen'] = grad_pen_loss.item()
        
        return metrics        

    def update(self, replay_iter, replay_iter_expert, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics
        
        batch = next(replay_iter)
        obs, action, reward_a, discount, next_obs = utils.to_torch(batch, self.device)
        
        batch_expert = next(replay_iter_expert)
        obs_e_raw, action_e, _, _, next_obs_e_raw = utils.to_torch(batch_expert, self.device)
        
        obs_e = self.aug(obs_e_raw.float())
        next_obs_e = self.aug(next_obs_e_raw.float())
        obs_a = self.aug(obs.float())
        next_obs_a = self.aug(next_obs.float())

        with torch.no_grad():
            obs_e = self.encoder(obs_e)
            next_obs_e = self.encoder(next_obs_e)
            obs_a = self.encoder(obs_a)
            next_obs_a = self.encoder(next_obs_a)

        # update critic
        if self.from_dem:
            metrics.update(self.update_discriminator(obs_a, action, obs_e, action_e))

            if self.RL_plus_IL:
                reward, metrics_r = self.compute_reward(obs, action, reward_a)
            else:
                reward, metrics_r = self.compute_reward(obs, action, reward_a=0)
        else:
            metrics.update(self.update_discriminator(obs_a, next_obs_a, obs_e, next_obs_e))

            if self.RL_plus_IL:
                reward, metrics_r = self.compute_reward(obs, next_obs, reward_a)
            else:
                reward, metrics_r = self.compute_reward(obs, next_obs, reward_a=0)

        metrics.update(metrics_r)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward_a.mean().item()

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics
