import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossTypeHGNNLayer(nn.Module):
    def __init__(self, in_features_list, out_features):
        super(CrossTypeHGNNLayer, self).__init__()
        # 为每种类型的节点定义不同的线性变换

        self.linear_layers = nn.ModuleList([nn.Linear(in_f, out_features) for in_f in in_features_list])

    def forward(self, node_features, cross_hypergraphs):
        num_types = len(node_features)
        updated_features = [torch.zeros_like(x) for x in node_features]

        for i in range(num_types):
            for j in range(num_types):
                if i == j:
                    continue
                H = cross_hypergraphs[i][j]

                # 计算每种节点类型的特征变换
                updated_features[i] += H @ node_features[j]

        # 将每种节点类型的特征映射到统一的输出空间
        return [self.linear_layers[i](updated_features[i]) for i in range(num_types)]

class CrossTypeHGNN(nn.Module):
    def __init__(self, dataset, hidden_features, out_features, node_types=3):
        super(CrossTypeHGNN, self).__init__()
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.HyperGraphInit(dataset,self.device)
        self.hgnn_layer1 = CrossTypeHGNNLayer(in_features_list, hidden_features)
        self.hgnn_layer2 = CrossTypeHGNNLayer([hidden_features] * node_types, out_features)
   
    def HyperGraphInit(self,dataset,device):
        user_ids = torch.tensor(dataset.inter_feat[dataset.uid_field] ,dtype=torch.int32).to(device)
        item_ids = torch.tensor(dataset.inter_feat[dataset.iid_field], dtype=torch.int32).to(device)
        
        category_ids=torch.tensor(dataset.item_feat["venue_category_id"],dtype=torch.int32).to(device)
        categories=category_ids[item_ids]#broadcast
        
        self.adjacency_tensor=torch.stack((user_ids,item_ids,categories),dim=1)

        Xs=torch.tensor(dataset.item_feat["x"],dtype=torch.float32).to(device)
        Ys=torch.tensor(dataset.item_feat["y"],dtype=torch.float32).to(device)
        
        item_Xs=Xs[item_ids]
        item_Ys=Ys[item_ids]
        
        
    
    def forward(self, node_features, cross_hypergraphs):
        hidden_features = self.hgnn_layer1(node_features, cross_hypergraphs)
        output_features = self.hgnn_layer2(hidden_features, cross_hypergraphs)
        return output_features

