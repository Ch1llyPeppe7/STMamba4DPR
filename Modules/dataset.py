from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.data.interaction import Interaction
import numpy as np
import torch
from torch.utils.data import BatchSampler,Sampler
import pandas as pd

import torch.nn.utils.rnn as rnn_utils
from recbole.utils import (
    FeatureSource,
    FeatureType,
    get_local_time,
    set_color,
    ensure_dir,
)
from Modules.myutils import *
from recbole.sampler import AbstractSampler
from collections import Counter
import copy
from scipy.spatial import KDTree

def unixTime2periodicVector(unitTime, components=None):
    dt = pd.to_datetime(unitTime, unit='s')

    # 定义时间特征和周期
    time_features = {
        'month': dt.month / 12,
        'weekday': dt.weekday() / 7,
        'day': dt.day / dt.days_in_month,
        'hour': dt.hour / 24,
        'minute': dt.minute / 60,
        'second': dt.second / 60
    }

    # 计算周期性编码
    periodic_vector = []
    if components is None:
        for feature in time_features.values():
            periodic_vector.append(np.sin(2 * np.pi * feature))
            periodic_vector.append(np.cos(2 * np.pi * feature))
    else:
        for component in components:
            feature = time_features[component]
            periodic_vector.append(np.sin(2 * np.pi * feature))
            periodic_vector.append(np.cos(2 * np.pi * feature))

    # 将返回值从 numpy 转为 torch.Tensor
    return torch.tensor(periodic_vector, dtype=torch.float32)

# 定义 FourSquare 类
class FourSquare(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)
        latitude = torch.tensor(self.item_feat['latitude'].values, dtype=torch.float32)
        longitude = torch.tensor(self.item_feat['longitude'].values, dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.item_feat['longitude'],self.item_feat['latitude']= self.lat_lon_to_spherical(latitude, longitude,device)
 
        
    def get_POI_KDTree(self):
        itemX=(self.item_feat["longitude"]).numpy()
        itemY=(self.item_feat["latitude"]).numpy()
        locations=np.stack((itemX,itemY),axis=-1)
        POI_tree=KDTree(locations)

    def lat_lon_to_spherical(self, latitudes, longitudes, device,radius=6371):
        """将经纬度转换为球面坐标 (x, y)"""
        # 将纬度和经度从度数转换为弧度
        lat_rad = latitudes.to(device) * (torch.pi / 180)  
        lon_rad = longitudes.to(device) * (torch.pi / 180)  
        
        x = radius * torch.cos(lat_rad) * torch.sin(lon_rad)*1000
        y = radius * torch.cos(lat_rad) * torch.cos(lon_rad)*1000
        
        
        min_x, min_y = x.min(), y.min()
    
        x -= min_x
        y -= min_y

        x = torch.round(x).to(torch.int32)
        y = torch.round(y).to(torch.int32)
        x,y=x.cpu(),y.cpu()
        torch.cuda.empty_cache()
        return x,y

  

  

class MySampler(AbstractSampler):
    """:class:`RepeatableSampler` is used to sample negative items for each input user. The difference from
    :class:`Sampler` is it can only sampling the items that have not appeared at all phases.

    Args:
        phases (str or list of str): All the phases of input.
        dataset (Dataset): The union of all datasets for each phase.
        distribution (str, optional): Distribution of the negative items. Defaults to 'uniform'.

    Attributes:
        phase (str): the phase of sampler. It will not be set until :meth:`set_phase` is called.
    """

    def __init__(self, phases, dataset, distribution="uniform", alpha=1.0):
        if not isinstance(phases, list):
            phases = [phases]
        self.phases = phases
        self.dataset = dataset

        self.iid_field = dataset.iid_field
        self.user_num = dataset.user_num
        self.item_num = dataset.item_num
        self.used_matrix()
        super().__init__(distribution=distribution, alpha=alpha)
    

    def used_matrix(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        POI_interaction_matrix,category_interaction_matrix,category_ids_counts,unique_IM=counting4all(self.dataset,device)
        itemX = self.dataset.item_feat["longitude"]
        itemY = self.dataset.item_feat["latitude"]

        center_X,center_Y,width,height=active_center_point(POI_interaction_matrix,unique_IM,itemX,itemY,device)
        catsim=category_interest_similarity(category_interaction_matrix,device)
        locsim=user_location_affinity_matrix(center_X,center_Y,width,height,device)

        #兴趣广度具有冷启动问题
        StrictCategory_threshold=0.9

        catsim_with_nan = catsim.clone()
        catsim_with_nan[catsim_with_nan == 0] = float('nan')
        QuantileCategory_threshold = torch.nanquantile(catsim_with_nan, 0.99, dim=1)#10 most similar user to filter

        inter_bool=(POI_interaction_matrix>0).float()

        mask=((locsim<1)*(locsim>=0.95)*(catsim>QuantileCategory_threshold)+((locsim==1)+(locsim==0))*(catsim>StrictCategory_threshold)).float()

        result=(mask@inter_bool)>0

        indices=result.nonzero(as_tuple=False)
                    
        temp_dict = {0: {0}}
        for key, value in indices.tolist():
            temp_dict.setdefault(key, set()).add(value)

        self.used_ids=list(temp_dict.values())
        #print((result.sum())/inter_bool.sum()) 潜在已知感兴趣对象增加量 

        

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.item_num, sample_num)

    def _get_candidates_list(self):
        return list(self.dataset.inter_feat[self.iid_field].numpy())

    def get_used_ids(self):
        """
        Returns:
            numpy.ndarray: Used item_ids is the same as positive item_ids.
            Index is user_id, and element is a set of item_ids.
        """
        return self.used_ids

    def sample_by_user_ids(self, user_ids, item_ids, num):
        """Sampling by user_ids.

        Args:
            user_ids (numpy.ndarray or list): Input user_ids.
            item_ids (numpy.ndarray or list): Input item_ids.
            num (int): Number of sampled item_ids for each user_id.

        Returns:
            torch.tensor: Sampled item_ids.
            item_ids[0], item_ids[len(user_ids)], item_ids[len(user_ids) * 2], ..., item_id[len(user_ids) * (num - 1)]
            is sampled for user_ids[0];
            item_ids[1], item_ids[len(user_ids) + 1], item_ids[len(user_ids) * 2 + 1], ...,
            item_id[len(user_ids) * (num - 1) + 1] is sampled for user_ids[1]; ...; and so on.
        """
        try:
            return self.sample_by_key_ids(user_ids, num)
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.user_num:
                    raise ValueError(f"user_id [{user_id}] not exist.")

    def sample_by_key_ids(self, key_ids, num):
        """Sampling by key_ids.

        Args:
            key_ids (numpy.ndarray or list): Input key_ids.
            num (int): Number of sampled value_ids for each key_id.

        Returns:
            torch.tensor: Sampled value_ids.
            value_ids[0], value_ids[len(key_ids)], value_ids[len(key_ids) * 2], ..., value_id[len(key_ids) * (num - 1)]
            is sampled for key_ids[0];
            value_ids[1], value_ids[len(key_ids) + 1], value_ids[len(key_ids) * 2 + 1], ...,
            value_id[len(key_ids) * (num - 1) + 1] is sampled for key_ids[1]; ...; and so on.
        """
        key_ids = np.array(key_ids)
        key_num = len(key_ids)
        total_num = key_num * num
        if (key_ids == key_ids[0]).all():
            key_id = key_ids[0]
            used = np.array(list(self.used_ids[key_id]))
            value_ids = self.sampling(total_num)
            check_list = np.arange(total_num)[np.isin(value_ids, used)]
            while len(check_list) > 0:
                value_ids[check_list] = value = self.sampling(len(check_list))
                mask = np.isin(value, used)
                check_list = check_list[mask]
        else:
            value_ids = np.zeros(total_num, dtype=np.int64)
            check_list = np.arange(total_num)
            key_ids = np.tile(key_ids, num)
            while len(check_list) > 0:
                value_ids[check_list] = self.sampling(len(check_list))
                check_list = np.array(
                    [
                        i
                        for i, used, v in zip(
                            check_list,
                           [self.used_ids[k] for k in key_ids[check_list]],
                            value_ids[check_list],
                        )
                        if v in used
                    ]
                )
        return torch.tensor(value_ids)


    def set_phase(self, phase):
        """Get the sampler of corresponding phase.

        Args:
            phase (str): The phase of new sampler.

        Returns:
            Sampler: the copy of this sampler, and :attr:`phase` is set the same as input phase.
        """
        if phase not in self.phases:
            raise ValueError(f"Phase [{phase}] not exist.")
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        return new_sampler
    
