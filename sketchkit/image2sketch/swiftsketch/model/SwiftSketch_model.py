import numpy as np
import torch
import torch.nn as nn
from ..utils.sketch_utils import get_features_dim


class SwiftSketch(nn.Module):
    def __init__(self,image_features_type= "CLIPMiddle_layer4",
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", normalize_model_output=0, 
                 cond_mode="no_cond", cond_mask_prob=0, arch='trans_dec',  emb_trans_dec=0, scaling_factor=2):
        super().__init__()


        self.arch= arch
        self.ncpoints = 4 #control points
        self.nfeats = 2 #x,y
        
        self.input_feats_dim = self.ncpoints * self.nfeats
        self.output_feats_dim= self.ncpoints * self.nfeats

        self.latent_dim = latent_dim
        self.image_features_dim = get_features_dim(image_features_type)

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
    
        self.normalize_output = normalize_model_output
        self.scaling_factor= scaling_factor
        self.cond_mode = cond_mode
        self.cond_mask_prob = cond_mask_prob

        self.input_process = InputProcess( self.input_feats_dim , self.latent_dim) #define linear layer 
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec
        
        if self.arch == 'trans_enc':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                            num_layers=self.num_layers)

        elif self.arch == 'trans_dec':
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)

            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers) 

        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec]')
                                                   
        

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond': #cond_mode = 'image'
            if self.arch == 'trans_enc' or (self.arch == 'trans_dec' and self.emb_trans_dec):
                self.embed_image = embed_image(self.image_features_dim, self.latent_dim,  cross_attention=False)
            if self.arch == 'trans_dec':
                self.embed_image_ca = embed_image(self.image_features_dim, self.latent_dim, cross_attention=True)
        

        self.output_process = OutputProcess(self.output_feats_dim, self.latent_dim, self.ncpoints,
                                            self.nfeats,self.normalize_output, self.scaling_factor) #define linear layer

       
    def parameters(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]


    def mask_cond(self, cond, force_mask=False): #for cfg
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.: 
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1,1,1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond



    def forward(self, x, timesteps, image_features=None, uncond=False, scale=None):
        """
        x: [batch_size, nstrokes, ncpoints, nfeats], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        x = self.input_process(x) #linear layer + reshape  [nstrokes, bs, d]

        emb = self.embed_timestep(timesteps)  # [1,bs, d]

        force_mask = uncond #for cfg 

        # create condition embed 

        if 'image' in self.cond_mode:
            enc_image= image_features
            if self.arch == 'trans_enc' or (self.arch == 'trans_dec' and self.emb_trans_dec):
                condition_emb= self.embed_image(self.mask_cond(enc_image, force_mask=force_mask)) #(BS, 512)
                # adding the image embed to the timestep embed
                emb += condition_emb  #[1, bs,d]

            if self.arch == 'trans_dec':
                condition_emb_ca= self.embed_image_ca(self.mask_cond(enc_image, force_mask=force_mask)) #(196, BS, 512)
                # adding the image embed to the timestep embed
                emb_ca = emb + condition_emb_ca  #[196, bs,d]

         
        if self.arch == 'trans_enc':
            # concatenate the conditions embed to the preprocessed input points 
            xseq = torch.cat((emb, x), axis=0)  # [nstrokes+1, bs, d]
            #PE
            xseq = self.sequence_pos_encoder(xseq)  # [nstrokes+1, bs, d] 
            output = self.seqTransEncoder(xseq)[1:]    # [nstrokes, bs, d]


        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)  # [nstrokes+1, bs, d]
            else:
                xseq = x #[nstrokes, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [nstrokes+1, bs, d]  or # [nstrokes, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb_ca)[1:] # [seqlen, bs, d]  
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb_ca)  
      

        output = self.output_process(output)  # [bs, ncpoints, nfeats, nstrokes]
        return output



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, input_feats_dim, latent_dim):
        super().__init__()
        self.input_feats_dim = input_feats_dim
        self.latent_dim = latent_dim
        self.pointsEmbedding = nn.Linear(self.input_feats_dim, self.latent_dim)


    def forward(self, x):
        # x shape [BS, nstrokes, 4, 2] 
        bs, nstrokes, ncpoints, nfeats = x.shape
        x = x.permute((1, 0, 2, 3)).reshape(nstrokes, bs, ncpoints*nfeats)
        x = self.pointsEmbedding(x)  # [nstrokes, bs, d]
        return x
     


class OutputProcess(nn.Module):
    def __init__(self, output_feats_dim, latent_dim, ncpoints, nfeats, normalize_output, scaling_factor):
        super().__init__()
        self.output_feats_dim = output_feats_dim
        self.latent_dim = latent_dim
        self.ncpoints = ncpoints
        self.nfeats = nfeats
        self.normalize_output = normalize_output
        self.scaling_factor= scaling_factor
        self.pointsFinal = nn.Linear(self.latent_dim, self.output_feats_dim)
      

    def forward(self, output):
        nstrokes, bs, d = output.shape
        output = self.pointsFinal(output)  # [nstrokes, bs, nfeats*nstrokes]
        if self.normalize_output :
            output = torch.tanh(output)  # Normalize to [-1, 1]
            output= output* self.scaling_factor
        
        output = output.reshape(nstrokes, bs, self.ncpoints, self.nfeats)
        output = output.permute(1, 0, 2, 3)  # [bs,nstrokes, ncpoints, nfeats]
        return output

        
    
def embed_image(image_features_dim, latent_dim, cross_attention):
    if cross_attention:
        model= CLIPMiddle_ca(image_features_dim, latent_dim) # output model dim [196, BS 512]
    else:
        model= CLIPMiddle(image_features_dim, latent_dim)  # output model dim [BS, 512]
    return model


class CLIPMiddle_ca(nn.Module): 
    def __init__(self, image_features_dim, latent_dim):
        super(CLIPMiddle_ca, self).__init__()
        input_dim, size = image_features_dim
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
       

    def forward(self, features):
        x = self.conv(features) # (BS,512,14,14)
        x = x.permute(2, 3, 0, 1).reshape(14 * 14, x.size(0), 512) #(14 * 14, BS, 512)
        x = self.fc(x)   #(14 * 14, BS, 512)
        return x
    
   
class CLIPMiddle(nn.Module):   
    def __init__(self, image_features_dim, latent_dim):
        super(CLIPMiddle, self).__init__()
        input_dim, size = image_features_dim
        self.fc = nn.Sequential(
            nn.Linear(256*size*size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, features):
        x = self.conv(features)
        x = x.reshape(x.size(0), -1) 
        x = self.fc(x)
        return x








