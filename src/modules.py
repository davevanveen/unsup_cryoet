import torch
import numpy as np
from functools import partial
from torch import nn
from collections import OrderedDict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import copy
from functools import partial


def sine_init(m, w0=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6/num_input)/w0, np.sqrt(6/num_input)/w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1/num_input, 1/num_input)


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def init_weights_uniform(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


class FirstSine(nn.Module):
    def __init__(self, w0=20):
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0*input)

            
def get_subdict(dictionary, key=None):
    ''' from torchmeta==1.4.0 modules.utils '''
    if dictionary is None:
        return None
    if (key is None) or (key == ''):
        return dictionary
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    return OrderedDict((key_re.sub(r'\1', k),value) for (k, value)
        in dictionary.items() if key_re.match(k) is not None)


class Sine(nn.Module):
    def __init__(self, w0=20):
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0*input)


class CoordinateNet(nn.Module):
    '''A canonical coordinate network'''
    def __init__(self, in_features, hidden_features, out_features,
                 num_hidden_layers=3, nl='sine',
                 w0=30, pe_scale=6, use_sigmoid=True, no_pe=False,
                 integrated_pe=False, **kwargs):

        super().__init__()

        self.nl = nl
        dims = in_features
        self.use_sigmoid = use_sigmoid
        self.no_pe = no_pe

        if integrated_pe:
            raise NotImplementedError('see mr_bacon repo for mip-nerf')

        if self.nl != 'sine' and not self.no_pe:
            in_features = hidden_features  # in_features * hidden_features

            self.pe = FFPositionalEncoding(hidden_features, pe_scale, dims=dims)

        self.net = FCBlock(in_features=in_features,
                           out_features=out_features,
                           num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features,
                           outermost_linear=True,
                           nonlinearity=nl,
                           w0=w0)

    def forward(self, model_input):  

        if isinstance(model_input, dict):
            coords = model_input['coords']
        else:
            coords = model_input

        if self.nl != 'sine' and not self.no_pe:
            coords_pe = self.pe(coords)
            output = self.net(coords_pe)
            if self.use_sigmoid:
                output = torch.sigmoid(output)
        else:
            output = self.net(coords)

        return {'model_in': model_input, 'model_out': {'output': output}}


class FFPositionalEncoding(nn.Module):
    ''' Fourier features positional encoding '''
    def __init__(self, embedding_size, scale, dims=2, gaussian=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale

        if gaussian:
            bvals = torch.randn(embedding_size // 2, dims) * scale 
        else:
            bvals = 2.**torch.linspace(0, scale, embedding_size//2) - 1

            if dims == 1:
                bvals = bvals[:, None]

            elif dims == 2:
                bvals = torch.stack([bvals, torch.zeros_like(bvals)], dim=-1)
                bvals = torch.cat([bvals, torch.roll(bvals, 1, -1)], dim=0)

            else:
                tmp = (dims-1)*(torch.zeros_like(bvals),)
                bvals = torch.stack([bvals, *tmp], dim=-1)

                tmp = [torch.roll(bvals, i, -1) for i in range(1, dims)]
                bvals = torch.cat([bvals, *tmp], dim=0)

        avals = torch.ones((bvals.shape[0]))
        self.avals = nn.Parameter(avals, requires_grad=False)
        self.bvals = nn.Parameter(bvals, requires_grad=False)

    def forward(self, tensor) -> torch.Tensor:
        """
            Apply positional encoding to the input.
        """

        return torch.cat([self.avals * torch.sin((2.*np.pi*tensor) @ self.bvals.T),
                          self.avals * torch.cos((2.*np.pi*tensor) @ self.bvals.T)], dim=-1)


def layer_factory(layer_type, w0=30):
    layer_dict = \
        {
         'relu': (nn.ReLU(inplace=True), init_weights_uniform),
         'sigmoid': (nn.Sigmoid(), None),
         'sine': (Sine(w0=w0), partial(sine_init, w0=w0)),
         'tanh': (nn.Tanh(), init_weights_xavier),
        }
    return layer_dict[layer_type]


class FCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''
    def __init__(self, in_features, out_features,
                 num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu',
                 weight_init=None, w0=30, set_bias=None,
                 dropout=0.0):
        super().__init__()

        self.first_layer_init = None
        self.dropout = dropout

        # Create hidden features list
        if not isinstance(hidden_features, list):
            num_hidden_features = hidden_features
            hidden_features = []
            for i in range(num_hidden_layers+1):
                hidden_features.append(num_hidden_features)
        else:
            num_hidden_layers = len(hidden_features)-1

        # Create the net
        #print(f"num_layers={len(hidden_features)}")
        if isinstance(nonlinearity, list):
            print(f"num_non_lin={len(nonlinearity)}")
            assert len(hidden_features) == len(nonlinearity), "Num hidden layers needs to " \
                                                              "match the length of the list of non-linearities"

            self.net = []
            self.net.append(nn.Sequential(
                nn.Linear(in_features, hidden_features[0]),
                layer_factory(nonlinearity[0])[0]
            ))
            for i in range(num_hidden_layers):
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[i], hidden_features[i+1]),
                    layer_factory(nonlinearity[i+1])[0]
                ))

            if outermost_linear:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                ))
            else:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                    layer_factory(nonlinearity[-1])[0]
                ))
        elif isinstance(nonlinearity, str):
            nl, weight_init = layer_factory(nonlinearity, w0=w0)
            if(nonlinearity == 'sine'):
                first_nl = FirstSine(w0=w0)
                self.first_layer_init = first_layer_sine_init
            else:
                first_nl = nl

            if weight_init is not None:
                self.weight_init = weight_init

            self.net = []
            self.net.append(nn.Sequential(
                nn.Linear(in_features, hidden_features[0]),
                first_nl
            ))

            for i in range(num_hidden_layers):
                if(self.dropout > 0):
                    self.net.append(nn.Dropout(self.dropout))
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[i], hidden_features[i+1]),
                    copy.deepcopy(nl)
                ))

            if (self.dropout > 0):
                self.net.append(nn.Dropout(self.dropout))
            if outermost_linear:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                ))
            else:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                    copy.deepcopy(nl)
                ))

        self.net = nn.Sequential(*self.net)

        if isinstance(nonlinearity, list):
            for layer_num, layer_name in enumerate(nonlinearity):
                self.net[layer_num].apply(layer_factory(layer_name, w0=w0)[1])
        elif isinstance(nonlinearity, str):
            if self.weight_init is not None:
                self.net.apply(self.weight_init)

            if self.first_layer_init is not None:
                self.net[0].apply(self.first_layer_init)

        if set_bias is not None:
            self.net[-1][0].bias.data = set_bias * torch.ones_like(self.net[-1][0].bias.data)

    def forward(self, coords):
        output = self.net(coords)
        return output