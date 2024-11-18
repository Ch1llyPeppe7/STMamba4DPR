import plotly.graph_objects as go
import random
import torch


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
    user_ids = torch.tensor(dataset.inter_feat[dataset.uid_field] ,dtype=torch.int32).to(device)
    item_ids = torch.tensor(dataset.inter_feat[dataset.iid_field], dtype=torch.int32).to(device)
    category_ids=torch.tensor(dataset.item_feat["venue_category_id"],dtype=torch.int32).to(device)
    categories=category_ids[item_ids]#broadcast

    num_users = user_ids.max()+1  #start from 1 padding row 0 with 0
    num_items = item_ids.max()+1  #start from 1 padding column 0 with 0
    num_category_ids=category_ids.max()+1 #start from 1 padding column 0 with 0

    POI_unique_indices = user_ids * num_items + item_ids
    category_unique_indices=user_ids*num_category_ids+categories

    POI_counts = torch.bincount(POI_unique_indices, minlength=num_users * num_items)
    POI_interaction_matrix = POI_counts.reshape(num_users, num_items)

    category_counts=torch.bincount(category_unique_indices, minlength=num_users * num_category_ids)
    category_interaction_matrix=category_counts.reshape(num_users,num_category_ids)

    category_ids_counts=torch.bincount(category_ids).to(dtype=torch.float64)
    category_ids_counts[0]=0#category0 for padding

    return POI_interaction_matrix,category_interaction_matrix,category_ids_counts

def kurtosis(data):
    # 计算均值和标准差
    mean = data.mean()
    std = data.std()

    # 标准化数据
    standardized_data = (data - mean) / std

    # 计算峰度，去掉 3（标准正态分布的峰度为 3）
    kurt = (standardized_data**4).mean() - 3
    return kurt



