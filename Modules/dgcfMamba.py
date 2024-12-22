import torch
from torch import nn
from Modules.MetaMamba import Mamba
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from Modules.myutils import *

class Mamba4POI(SequentialRecommender):
    def __init__(self, config, dataset):
        super(Mamba4POI, self).__init__(config, dataset)
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.n_factors=config["n_factors"]
        # Hyperparameters for Mamba block
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]
        self.TIME_SEQ_FIELD=config['TIME_FIELD']+config['LIST_SUFFIX']
        self.locdim=int(self.hidden_size/2/self.n_factors)
        self.catdim=int(self.hidden_size/2)

        self.itemloc_embedding,self.itemcat_embedding,self.userloc_embedding,self.usercat_embedding = self._init_embedding(dataset)

        self.itembase_embedding = nn.Embedding(
            self.num_items, self.locdim, padding_idx=0
        )
        self.userbase_embedding = nn.Embedding(
            self.num_users, self.locdim, padding_idx=0
        )
 
        self.LocNorm = nn.LayerNorm(self.locdim, eps=1e-12)  # 针对地理位置编码
        self.Norm = nn.LayerNorm(self.catdim, eps=1e-12)  # 针对地理位置编码
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.mamba1 = nn.ModuleList([
            MambaLayer(
                d_model=self.locdim,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])
        self.mamba2 = nn.ModuleList([
            MambaLayer(
                d_model=self.locdim,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])
        self.mamba3 = nn.ModuleList([
            MambaLayer(
                d_model=self.locdim,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])
        self.mamba4 = nn.ModuleList([
            MambaLayer(
                d_model=self.locdim,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])
        
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

    def item_embedding(self,item_seq=None):
        if item_seq is None:
            item_seq = torch.arange(self.num_items, device=self.device)

        return self.itemloc_embedding(item_seq)+self.itembase_embedding(item_seq)
    
    def user_embedding(self,user_seq=None):
        if user_seq is None:
            user_seq = torch.arange(self.num_users, device=self.device)

        return self.userloc_embedding(user_seq)+self.userbase_embedding(user_seq)
           

    def _init_embedding(self,dataset):
        num_category=(dataset.item_feat["venue_category_id"].max()+1).to(torch.int)
        num_user=(dataset.inter_feat["user_id"].max()+1).to(torch.int)
        IM,M0,_,Uim=counting4all(dataset,self.device)

        Ec=torch.zeros((num_category,self.catdim),dtype=torch.float32)
        Eu=torch.zeros((num_user,self.catdim),dtype=torch.float32)
        row_nonzero_mask=M0.sum(1)>0
        col_nonzero_mask=M0.sum(0)>0
        nonzero_M0=M0[row_nonzero_mask][:,col_nonzero_mask]
        
        E0u,S0,E0c=UC_SVD(nonzero_M0,self.catdim,self.device)

        Ec[col_nonzero_mask]=E0c
        Eu[row_nonzero_mask]=E0u

        itemX=dataset.item_feat["longitude"]
        itemY=dataset.item_feat["latitude"]
        itemC=dataset.item_feat["venue_category_id"]
        self.itemX=itemX.clone().to(self.device)
        self.itemY=itemY.clone().to(self.device)
        self.itemC=itemC.clone().to(self.device)
       
        Locations=torch.stack([itemX,itemY],dim=1)
        centerx,centery,_,_=active_center_point(IM,Uim,itemX,itemY,self.device)
        
        UsrLocations=torch.stack([centerx,centery],dim=1)
        
        ItemLocEnco=sinusoidal_position_encoding(Locations,self.locdim,self.device)
        UsrLocEnco=sinusoidal_position_encoding(UsrLocations,self.locdim,self.device)

        itemcat_embedding = nn.Embedding.from_pretrained(
            Ec[itemC], freeze=False
        )

        usercat_embedding = nn.Embedding.from_pretrained(
            Eu, freeze=False
        )

        itemloc_embedding = nn.Embedding.from_pretrained(
            ItemLocEnco, freeze=True
        )

        userloc_embedding = nn.Embedding.from_pretrained(
            UsrLocEnco, freeze=True
        )

        self.num_items=ItemLocEnco.shape[0]
        self.num_users=num_user
        return itemloc_embedding,itemcat_embedding,userloc_embedding,usercat_embedding


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len,user_list):
        LocState=self.LocNorm(self.itemloc_embedding(item_seq))
        EmbState=self.Norm(self.itembase_embedding(item_seq))
       
        UsrBase=self.Norm(self.userbase_embedding(user_list))
        UsrLoc=self.Norm(self.userloc_embedding(user_list))
        Start=UsrBase.unsqueeze(1)+LocState[:,0,:]
        
        ItemState=LocState.repeat(1,1,4)+EmbState

        State=ItemState#torch.concat((Start,ItemState),dim=1)
        n_seg=LocState.shape[1]
        for i in range(self.num_layers):
            State[:,:,:n_seg] = self.mamba1[i](State[:,:,:n_seg])
            State[:,:,n_seg:2*n_seg] = self.mamba2[i](State[:,:,n_seg:2*n_seg])
            State[:,:,2*n_seg:3*n_seg] = self.mamba3[i](State[:,:,2*n_seg:3*n_seg])
            State[:,:,3*n_seg:] = self.mamba3[i](State[:,:,3*n_seg:])

        Seq_output=self.gather_indexes(State, item_seq_len - 1)
        return Seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_list=interaction[self.USER_ID]
        item_output = self.forward(item_seq, item_seq_len,user_list)
        UsrBase=self.Norm(self.userbase_embedding(user_list))
        pos_items = interaction[self.POS_ITEM_ID]
    

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.mean(item_output * pos_items_emb, dim=-1) +torch.mean(item_output * pos_items_emb, dim=-1)  # [B]
            neg_score =torch.mean(item_output * neg_items_emb, dim=-1)+torch.mean(item_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding()
            logits = torch.matmul(item_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)+self.reward(item_output,UsrBase)
        torch.cuda.empty_cache()    
        return loss
       
    def reward(self,item_output,UsrBase):
        return -torch.mean(item_output * UsrBase, dim=-1) 

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        user_list=interaction[self.USER_ID]
        UsrBase=self.Norm(self.userbase_embedding(user_list))
        item_output = self.forward(item_seq, item_seq_len,user_list)
    
    
        test_item_emb = self.item_embedding(test_item)
        
        scores = torch.mul(item_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_list=interaction[self.USER_ID]

        item_output = self.forward(item_seq, item_seq_len,user_list)
        

        test_items_emb = self.item_embedding()
        scores = torch.matmul(
            item_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
    


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model*4, dropout=dropout)
    
    def forward(self, input_tensor):
        # 检查 input_tensor 维度，并添加 batch 维度
        if input_tensor.dim() == 2:  # 例如 [seq_len, d_model]
            input_tensor = input_tensor.unsqueeze(0)  # 添加 batch 维度 [1, seq_len, d_model]
        elif input_tensor.dim() == 1:  # 单个序列
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # 变为 [1, 1, d_model]

        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)

        hidden_states = self.ffn(hidden_states)
        return hidden_states
    
class ReverseMessagePassing(nn.Module):
    def __init__(self,catdim,locdim):
        super().__init__()
        self.catdim=catdim
        self.locdim=locdim
    def forward(self, E):
        KQ=torch.einsum('bld,bld->bl1', E[:,:,:-1], E[:,:,1:])
    
        return 


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
    
    