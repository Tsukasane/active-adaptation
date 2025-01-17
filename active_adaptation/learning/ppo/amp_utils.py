from tensordict import TensorDict
import glob
import joblib
import torch
import torch.nn as nn

class AMPLoader:
    def __init__(self, data_dir, amp_length, amp_key, device):
        self.data_dir = data_dir
        self.amp_length = amp_length
        self.key = amp_key
        self.device = device
        
        self.data = self.load_data()
        self.segments = self.pre_slice()

    def load_data(self):
        data = []
        for file in glob.glob(f"{self.data_dir}/*/*.pkl"):
            trajectory = joblib.load(file)
            body_pose = torch.tensor(trajectory["keypoints"], dtype=torch.float32)
            joint_pos = torch.tensor(trajectory["qpos"], dtype=torch.float32)
            linear_vel = torch.tensor(trajectory["root_linear_velocity"], dtype=torch.float32)
            angular_vel = torch.tensor(trajectory["root_angular_velocity"], dtype=torch.float32)
            amp_data = torch.cat([body_pose, joint_pos, linear_vel, angular_vel], dim=-1) # [T, 12 * 3 + 23 + 3 + 3]
            data.append(amp_data)
        data = torch.cat(data, dim=0)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def pre_slice(self):
        N = self.data.shape[0]
        M = N - self.amp_length + 1
        indices = torch.arange(M).unsqueeze(1) + torch.arange(self.amp_length).unsqueeze(0)
        segments = self.data[indices]       # [M, amp_length, feature]
        return segments
    
    def sample_batch(self, batch_size):
        seg_idx = torch.randint(0, len(self.segments), (batch_size,))
        amp_batch = self.segments[seg_idx].to(self.device)     # [B, amp_length, 12 * 3 + 23 + 3 + 3]
        tensordict = TensorDict()
        tensordict.set(self.key, amp_batch.reshape(batch_size, -1))
        return tensordict
    
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_sizes, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.model = nn.Sequential()
        self.amp_linear = nn.Linear(hidden_sizes[-1], 1)
        for i, (in_size, out_size) in enumerate(zip([input_size] + hidden_sizes[:-1], hidden_sizes)):
            self.model.add_module(f"fc{i}", nn.Linear(in_size, out_size))
            self.model.add_module(f"layer_norm{i}", nn.LayerNorm(out_size))
            self.model.add_module(f"mish{i}", nn.Mish())
        self.model.add_module(f"fc{len(hidden_sizes)}", self.amp_linear)
        self.model.to(self.device)
        
    def forward(self, x):
        return self.model(x)
    
    # Compute the regularization term
    def compute_disc_logit_reg(self, disc_logit_reg=0.01):
        return disc_logit_reg * torch.sum(torch.square(self.amp_linear.weight))
    
    # wasserstein gradient penalty
    def gradient_penalty(self, expert_data):
        expert_data.requires_grad = True
        disc_interpolates = self(expert_data)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=expert_data,
            grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 0) ** 2).mean()
        return gradient_penalty
    
    def amp_reward(self, trajs):
        d = self.model(trajs)
        style_reward = torch.clamp(1-(1/4)*torch.square(d-1), min=0.)
        return style_reward

