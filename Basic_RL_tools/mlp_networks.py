import torch
from torch import nn
from torch.nn import functional as F
import pytorch_util as ptu
# from torchvision.models import resnet18, ResNet18_Weights

def identity(x):
    return x

class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """
    def __init__(self, features, center=True, scale=False, eps=1e-6):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output


class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output



class ConcatMlp(Mlp):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)


class ResNet_MLP(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            embedding_dim,
            self_perception_dim,
            action_dim,
            output_size,
            next_n_step,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__()
        self.self_perception_dim = self_perception_dim
        self.action_dim = action_dim
        self.next_n_step = next_n_step
        self.ResNet = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False) # if you want to block, use .eval()
        self.res_embedding = nn.Linear(1000, embedding_dim)
        self.MLP = Mlp(
            hidden_sizes,
            input_size=embedding_dim * (next_n_step + 1) + self_perception_dim + action_dim,
            output_size=output_size,
            init_w=init_w,
            **kwargs
        )


    def forward(self, image, obs, action):
        batch_size = image.shape[0]
        h = self.ResNet(image.view(-1, 3, 224, 224))
        h = self.res_embedding(h).view(batch_size, -1) # [N, embedding_dim * 4]
        h = torch.cat((h, obs, action), dim=1)
        h = self.MLP(h)
        return h


class CNN_MLP(nn.Module):
    def __init__(
        self,
        hidden_sizes,
        feature_dim,
        self_perception_dim,
        action_dim,
        output_size,
        next_n_step,
        image_width,
        image_height,
        init_w=1e-3,
        **kwargs
    ):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.feature_dim = feature_dim
        self.self_perception_dim = self_perception_dim
        self.action_dim = action_dim
        self.next_n_step = next_n_step
        self.CNN = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5184, 512),
            nn.ReLU(),
            nn.Linear(512, self.feature_dim),
        )
        self.MLP = Mlp(
            hidden_sizes,
            input_size=feature_dim * (next_n_step + 1) + self_perception_dim + action_dim,
            output_size=output_size,
            init_w=init_w,
            **kwargs
        )

    def forward(self, image, obs, action):
        batch_size = image.shape[0]
        h = self.CNN(image.view(-1, 3, self.image_width, self.image_height)).view(batch_size, -1) # [N, embedding_dim * 4]
        h = torch.cat((h, obs, action), dim=1)
        h = self.MLP(h)
        return h



# class CNN_feature0(nn.Module):
#     def __init__(
#         self,
#         hidden_sizes,
#         feature_dim,
#         image_width,
#         image_height,
#         init_w=1e-3,
#         **kwargs
#     ):
#         super().__init__()
#         self.image_width = image_width
#         self.image_height = image_height
#         self.feature_dim = feature_dim
#         self.hidden_sizes = hidden_sizes
#         self.CNN = nn.Sequential(
#             nn.Conv2d(3, 32, 7, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 5, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 5),
#             nn.ReLU(),
#             nn.Flatten(-3),
#             nn.Linear(5184, hidden_sizes[0]),
#             nn.ReLU(),
#             nn.Linear(hidden_sizes[0], hidden_sizes[1]),
#             nn.ReLU(),
#             nn.Linear(hidden_sizes[1], self.feature_dim),
#             # nn.tanh()
#         )


#     def forward(self, image):
#         h = self.CNN(image) # self,feature_dim
#         return h



class CNN_feature(nn.Module):
    def __init__(self,             
            state_dim,
            action_dim,
            img_channel,
            img_width,
            img_height,
            image_feature_dim,
            hidden_width,
            ):
        super(CNN_feature, self).__init__(
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.img_channel = img_channel
        self.img_width = img_width
        self.img_height = img_height
        self.image_feature_dim = image_feature_dim
        self.hidden_width = hidden_width
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, 3, self.img_width, self.img_height)).shape[1]

        self.cnn_linear = nn.Sequential(
            nn.Linear(n_flatten, self.image_feature_dim),
            nn.ReLU()
        )

    def preprocessing(self, img):
        return img/ 255.0 *2 -1

    def forward(self, Img):
        Img = self.preprocessing(Img)
        h = self.cnn(Img) 
        h = self.cnn_linear(h)
        return h