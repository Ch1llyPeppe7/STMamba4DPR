# @Time   : 2024/11/24
# @Author : Jin Qian
# @Email  : chillypepper@foxmail.com
import plotly.graph_objects as go
import random
import torch

def accumulate_category(Cs,userids,UC):
    Cs_tensor = torch.tensor(Cs, dtype=torch.int32)
    userids_tensor = torch.tensor(userids, dtype=torch.int32)

    combined_tensor = torch.stack((Cs_tensor, userids_tensor), dim=1)
    for cid,uid in combined_tensor:
        UC.index_put_((uid,cid),torch.tensor(1,dtype=torch.float32),accumulate=True)
    return UC

def UC_SVD(M,k,device):
    normalize_M=(M/M.sum(1,keepdim=True)).to(device)

    U, S, Vt = torch.linalg.svd(normalize_M, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    return U_k.cpu(),S_k.cpu(),Vt_k.T.cpu()


def sinusoidal_position_encoding(locations,locdim,device):
    locations=(locations/200).to(device)#<1km dot>0.4
    d=int(locdim/2)
    print(d)
    position_encoding = torch.zeros((locations.shape[0],locdim)).to(device)
    for i in range(0, d, 2):
        position_encoding[:, i] = torch.sin(locations[:,0] * (-i / d)**3)
        position_encoding[:,i + 1] = torch.cos(locations[:,0] * (-(i + 1) / d)**3)
        position_encoding[:, d+i] = torch.sin(locations[:,1] * (-i / d)**3)
        position_encoding[:,d+i + 1] = torch.cos(locations[:,1] *(-(i + 1) / d)**3)

    return position_encoding.cpu()

def user_location_affinity_matrix(center_X,center_Y,width,height,device):
    #不同用户对同一用户的相似度在同一尺度上 
    xmax=center_X+width
    xmin=center_X-width
    ymax=center_Y+height
    ymin=center_Y-height

    area=4*width*height
    #UnionArea = area.unsqueeze(1) + area.unsqueeze(0)

    crossXmax=torch.min(xmax.unsqueeze(1),xmax.unsqueeze(0))
    crossXmin=torch.max(xmin.unsqueeze(1),xmin.unsqueeze(0))

    crossYmax=torch.min(ymax.unsqueeze(1),ymax.unsqueeze(0))
    crossYmin=torch.max(ymin.unsqueeze(1),ymin.unsqueeze(0))
    
    intersection_width = torch.clamp(crossXmax - crossXmin, min=0)
    intersection_height = torch.clamp(crossYmax - crossYmin, min=0)
    intersection_area = intersection_width * intersection_height
  
    #sim=intersection_area/(UnionArea-intersection_area+((UnionArea-intersection_area)==0).float())
    sim=intersection_area/((area+(area==0).float()).unsqueeze(1))
    
    return sim

def active_center_point(interaction_matrix,uniqueIM,itemX,itemY,device):
    torch.cuda.empty_cache()
    itemX=itemX.to(device)
    itemY=itemY.to(device)
    interaction_matrix=interaction_matrix.to(device)
    uniqueIM=uniqueIM.to(device)
    
    X=(uniqueIM*itemX)
    Y=(uniqueIM*itemY)
    Cx=X.sum(1)/((X>0).float().sum(1)+((X>0).float().sum(1)==0).float())
    Cy=Y.sum(1)/((Y>0).float().sum(1)+((Y>0).float().sum(1)==0).float())

    weight=interaction_matrix/(interaction_matrix+(interaction_matrix==0).float()).sum(dim=1,keepdim=True)

    center_x,center_y=Cx+(weight*X).sum(1),Cy+(weight*Y).sum(1)

    dX=torch.abs(X-center_x.unsqueeze(1))
    dY=torch.abs(Y-center_y.unsqueeze(1))

    width=dX.max(1)[0].cpu()
    height=dY.max(1)[0].cpu()
  
    
    return center_x.cpu(),center_y.cpu(),width,height



def category_interest_similarity(category_interaction_matrix,device):
    #无法解决冷启动问题 嵌入向量小 找不到相似用户
    CatMat=category_interaction_matrix.double().to(device)
    rowsum=CatMat.sum(dim=1,keepdim=True)#distribution matrix
    DM=CatMat/(rowsum+(rowsum==0).double())
    norm=torch.sqrt((DM*DM).sum(1))
    norm2=norm.view(-1,1)@norm.view(1,-1)#cosine similarity which measure the similarity of the shape
    sim=DM@DM.T/(norm2+(norm2==0).double())
    sim.fill_diagonal_(1)
    return sim.cpu()


def visualize_3d_hypergraph(HyperEdge, user_ids=None, user_num=1):
    # 确保随机选择指定数量的用户
    unique_user_ids = torch.unique(HyperEdge[:, 0]).cpu().numpy()
    if user_ids is None:
        user_ids = random.sample(list(unique_user_ids), min(user_num, len(unique_user_ids)))
    
    # 准备所有节点和超边
    all_user_nodes = []
    all_item_nodes = []
    all_category_nodes = []
    all_hyperedges = []

    # 遍历选择的用户
    for user_id in user_ids:
        # 提取与该用户相关的超边
        user_edges = HyperEdge[HyperEdge[:, 0] == user_id, :]
        item_ids = user_edges[:, 1].cpu().numpy()
        categories = user_edges[:, 2].cpu().numpy()

        # 创建当前用户节点
        user_node = f"User {user_id}"
        item_nodes = [f"Item {item_id}" for item_id in item_ids]
        category_nodes = [f"Category {category}" for category in categories]

        all_user_nodes.append(user_node)
        all_item_nodes.extend(item_nodes)
        all_category_nodes.extend(category_nodes)

        # 构建用户的超边
        for item_node, category_node in zip(item_nodes, category_nodes):
            all_hyperedges.append((user_node, category_node, item_node))

    # 去重所有节点
    unique_user_nodes = list(set(all_user_nodes))
    unique_item_nodes = list(set(all_item_nodes))
    unique_category_nodes = list(set(all_category_nodes))
    unique_nodes = unique_user_nodes + unique_category_nodes + unique_item_nodes

    # 创建节点索引
    node_indices = {node: idx for idx, node in enumerate(unique_nodes)}

    # 层次化坐标分布
    z_positions = {
        "user": 1.0,  # 用户层最高
        "category": 0.5,  # 类别层
        "item": 0.0  # 物品层
    }

    # 三维布局
    positions = {}
    for idx, user_node in enumerate(unique_user_nodes):
        positions[user_node] = (random.random(), random.random(), z_positions["user"])
    for idx, category_node in enumerate(unique_category_nodes):
        positions[category_node] = (random.random(), random.random(), z_positions["category"])
    for idx, item_node in enumerate(unique_item_nodes):
        positions[item_node] = (random.random(), random.random(), z_positions["item"])

    # 准备节点绘制
    node_x = [positions[node][0] for node in unique_nodes]
    node_y = [positions[node][1] for node in unique_nodes]
    node_z = [positions[node][2] for node in unique_nodes]
    node_colors = (
        ["blue"] * len(unique_user_nodes) +  # 用户为蓝色
        ["orange"] * len(unique_category_nodes) +  # 类别为橙色
        ["green"] * len(unique_item_nodes)  # 物品为绿色
    )

    # 绘制边
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in all_hyperedges:
        x_coords = [positions[edge[0]][0], positions[edge[1]][0], positions[edge[2]][0], positions[edge[0]][0]]
        y_coords = [positions[edge[0]][1], positions[edge[1]][1], positions[edge[2]][1], positions[edge[0]][1]]
        z_coords = [positions[edge[0]][2], positions[edge[1]][2], positions[edge[2]][2], positions[edge[0]][2]]
        edge_x.extend(x_coords + [None])
        edge_y.extend(y_coords + [None])
        edge_z.extend(z_coords + [None])

    # 创建图形
    fig = go.Figure()

    # 添加边
    fig.add_trace(go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(color='gray', width=1),
        hoverinfo='none'
    ))

    # 添加节点
    fig.add_trace(go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode='markers+text',
        marker=dict(size=10, color=node_colors),
        text=unique_nodes,
        textposition="top center",
        hoverinfo='text'
    ))

    # 更新布局
    fig.update_layout(
        title=f"3D Hypergraph for {len(user_ids)} Users",
        scene=dict(
            xaxis=dict(title="X Axis", showgrid=False, zeroline=False),
            yaxis=dict(title="Y Axis", showgrid=False, zeroline=False),
            zaxis=dict(title="Z Axis", showgrid=False, zeroline=False),
        ),
        showlegend=False
    )

    fig.show("browser")


    
def reciprocal_rarity(category_ids_counts,dim=0):
    '''
    parameters:
        category_ids_counts:tensor quantity of each category for whole users or single user
        dim:int set dim=1 when category_ids_counts based on whole users
    notes: This methods is for test and comparation
    '''
    reciprocal=1/category_ids_counts
    reciprocal[0]=0#modify inf
    return reciprocal/reciprocal.sum(dim)


def exp_rarity(category_ids_counts,device,dim=0):
    '''
    parameters:
        category_ids_counts:tensor quantity of each category for whole users or single user
        dim:int set dim=1 when category_ids_counts based on whole users
    notes:
        We use exp to modify the distribution of category quantity,which higly refine 
    the discrimination of the rarity.We proposed scaling the rich/frequently visited 
    categories into the slow-changing area while the rare are highly discriminated.    
    '''
    #zoom control hyperparameters
    e1=torch.tensor(0.3,dtype=torch.float64).to(device) #derivative for start point
    e2=torch.tensor(1,dtype=torch.float64).to(device)  #second derivative for boost point
    boost=torch.tensor(3,dtype=torch.float64).to(device)
    a=torch.exp(boost).to(device)
    p1=torch.log(e1/torch.log(a)/torch.log(a))/torch.log(a)
    p2=torch.log(e2/torch.log(a)/torch.log(a))/torch.log(a)

    #p1-p2 goes slow after p2 it boost!
    #compress mins-max into d1-d2 while put mins after e2 which arouse mutaion to distinguish them

    quant=torch.tensor(0.9,dtype=torch.float64).to(device)
    quantile=torch.quantile(category_ids_counts,quant,dim=dim)

    C=(category_ids_counts.max(dim=dim).values-category_ids_counts)*(p2-p1)/(category_ids_counts.max(dim=dim).values-quantile)+p1
    #start from p1 boost after p2

    exp=torch.exp(torch.log(a)*C)
    exp[0]=0
    return exp/exp.sum(dim=dim)

def counting4all(dataset,device):
    '''
    parameters:
        self:base_sampler offering the whole dataset
        device:cuda calculation platform
    return:
        POI_interaction_matrix     :interactions based on specific places
        category_interaction_matrix:interactions based on categories
        category_ids_count         :inherent features of POI based on categories 
                                    reflected by the quantity distribution of which,
                                    are to generate the probablity in view of rarity.
    '''
    user_ids = dataset.inter_feat[dataset.uid_field].to(device)
    item_ids = dataset.inter_feat[dataset.iid_field].to(device)

    category_ids=dataset.item_feat["venue_category_id"].to(device)
    categories=category_ids[item_ids]#broadcast

    num_users = user_ids.max()+1  #start from 1 padding row 0 with 0
    num_items = dataset.item_feat[dataset.iid_field].max()+1  #start from 1 padding column 0 with 0
    num_category_ids=category_ids.max()+1 #start from 1 padding column 0 with 0

    
    unique_POI_IM = torch.zeros(num_users, num_items, dtype=torch.float32)
    unique_POI_IM[user_ids, item_ids] = 1 

    POI_unique_indices = user_ids * num_items + item_ids
    category_unique_indices=user_ids*num_category_ids+categories

    POI_counts = torch.bincount(POI_unique_indices, minlength=num_users * num_items)
    POI_interaction_matrix = POI_counts.reshape(num_users, num_items).cpu()

    category_counts=torch.bincount(category_unique_indices, minlength=num_users * num_category_ids)
    category_interaction_matrix=category_counts.reshape(num_users,num_category_ids).cpu()

    category_ids_counts=torch.bincount(category_ids).to(dtype=torch.float64).cpu()
    category_ids_counts[0]=0#category0 for padding
    torch.cuda.empty_cache()
    return POI_interaction_matrix,category_interaction_matrix,category_ids_counts,unique_POI_IM

def kurtosis(data):
    # 计算均值和标准差
    mean = data.mean()
    std = data.std()

    # 标准化数据
    standardized_data = (data - mean) / std

    # 计算峰度，去掉 3（标准正态分布的峰度为 3）
    kurt = (standardized_data**4).mean() - 3
    return kurt



