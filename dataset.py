from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.data.interaction import Interaction
import numpy as np
import torch
import pandas as pd
from logging import getLogger
import torch.nn.utils.rnn as rnn_utils
from recbole.utils import (
    FeatureSource,
    FeatureType,
    get_local_time,
    set_color,
    ensure_dir,
)
from myutils import *
from recbole.data.dataloader.abstract_dataloader import (
    AbstractDataLoader,NegSampleDataLoader
)
from recbole.utils import InputType, FeatureType, FeatureSource, ModelType

from numpy.random import sample
from collections import Counter
import copy


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
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.field2seqlen = {
        getattr(self,f"{self.iid_field}_list_field", None): self.max_seq_length,
            'time_encoded': 12  # time_encoded 是一个长度为 12 的向量
        }
        self.set_field_property('latitude', FeatureType.FLOAT, FeatureSource.ITEM, 1)  
        self.set_field_property('longtitude', FeatureType.FLOAT, FeatureSource.ITEM, 1) 


        
    def _change_feat_format(self):
        # 获取经纬度数据，并转换为 torch 张量 
        latitude = torch.tensor(self.item_feat['latitude'].values, dtype=torch.float32)
        longitude = torch.tensor(self.item_feat['longitude'].values, dtype=torch.float32)

        # 如果 GPU 可用，将数据移到 GPU 上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            

    

        self.item_feat['longtitude'],self.item_feat['latitude']= self.lat_lon_to_spherical(latitude, longitude,device)
        

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

  

    def copy_field_property(self, dest_field, source_field):
        """Copy properties from ``dest_field`` towards ``source_field``.

        Args:
            dest_field (str): Destination field.
            source_field (str): Source field.
        """
        self.field2type[dest_field] = self.field2type[source_field]
        self.field2source[dest_field] = self.field2source[source_field]
        self.field2seqlen[dest_field] = self.field2seqlen.get(source_field, 0)
         


class MyTrainDataLoader(NegSampleDataLoader):
    def __init__(self, config, dataset, sampler, shuffle):
        self.sample_size = len(dataset)
        self.batch_size =config["train_batch_size"] 
        del dataset.inter_feat['timestamp']#去除冗余
        super().__init__(config, dataset, sampler, shuffle)
        super()._set_neg_sample_args(config, dataset, InputType.PAIRWISE, config["train_neg_sample_args"])
    

    def _init_batch_size_and_step(self):
        self.step = self.batch_size 

    def collate_fn(self, index):
        return super()._neg_sampling(self._dataset.inter_feat[index])





        
class MyValidDataLoader(NegSampleDataLoader):
    def __init__(self, config, dataset,train_dataset, sampler, shuffle):
        # # 获取样本大小和初始化参数
        self.sample_size = len(dataset)
        self.batch_size =config["eval_batch_size"] 
        # self.hist_item=train_dataset.inter_feat[train_dataset.iid_field]
        # self.hist_user=train_dataset.inter_feat[train_dataset.uid_field]
        super().__init__(config, dataset, sampler, shuffle)
        super()._set_neg_sample_args(config, dataset, InputType.PAIRWISE, config["valid_neg_sample_args"])
        del self._dataset.inter_feat['timestamp']
    def _init_batch_size_and_step(self):
        self.step = self.batch_size 

    def collate_fn(self, index):
        data=super()._neg_sampling(self._dataset.inter_feat[index])
        

        num_batch=item_id_list.size(0)
        for idx in range(num_batch):
           hist_idx = (self.hist_user == user_id[idx]).nonzero(as_tuple=True)[0]  # 获取当前 idx 对应的匹配索引
           batch_idx = torch.full((hist_idx.size(0), 1), idx)  # 创建一个大小为 (N, 1) 的张量，值全为 idx
           hist_idx = torch.cat((batch_idx, hist_idx.unsqueeze(1)), dim=1)  # 将 idx 加到每个元素前

        positive_i=self._dataset.inter_feat[self._dataset.iid_field][valid_indices]
        
        first_occurrence = {}
        for i, uid in enumerate(user_id):
            # 使用 .item() 将 tensor 转换为标量值
            if uid.item() not in first_occurrence:
                first_occurrence[uid.item()] = i

        # 使用 .item() 将每个 uid 转换为普通的数值类型
        positive_u = torch.tensor([first_occurrence[uid.item()] for uid in user_id], dtype=torch.long)

        return [Interaction(data),hist_idx,positive_u,positive_i]

    # def update_config(self, config):
    #     """
    #     更新配置，并重新初始化批次大小和步长。
    #     """
    #     super().update_config(config)
    #     self._init_batch_size_and_step()


class MyTestDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset,train_dataset, sampler, shuffle):
        # 获取样本大小和初始化参数
        self.sample_size = len(dataset)
        self._sampler = sampler 
        self.batch_size =config["eval_batch_size"] 
        self._init_batch_size_and_step()
        self.hist_item=train_dataset.inter_feat[train_dataset.iid_field]
        self.hist_user=train_dataset.inter_feat[train_dataset.uid_field]
        super().__init__(config, dataset, sampler, shuffle)
    
    def _init_batch_size_and_step(self):
        self.step = self.batch_size 

    def collate_fn(self, index):
        # 将索引数组转换为一个批次的数据
        index = np.array(index)
        # 提取测试批次数据
        valid_indices = index[index != -1]
        # 直接从 inter_feat 中提取批次数据
        user_id=self._dataset.inter_feat[self._dataset.uid_field][valid_indices]
        item_id_list=self._dataset.inter_feat[self._dataset.iid_field][valid_indices].unsqueeze(0)
        item_length = torch.tensor([len(item_id_list[0]) - 1], dtype=torch.long)
        pos_item=(item_id_list[:,item_length-1])
        data = {
            self._dataset.uid_field: user_id,
            self._dataset.ITEM_SEQ: item_id_list,
            self._dataset.ITEM_SEQ_LEN: item_length,
            self._dataset.iid_field: pos_item 
        }
        num_batch=item_id_list.size(0)
        for idx in range(num_batch):
           hist_idx = (self.hist_user == user_id[idx]).nonzero(as_tuple=True)[0]  # 获取当前 idx 对应的匹配索引
           batch_idx = torch.full((hist_idx.size(0), 1), idx)  # 创建一个大小为 (N, 1) 的张量，值全为 idx
           hist_idx = torch.cat((batch_idx, hist_idx.unsqueeze(1)), dim=1)  # 将 idx 加到每个元素前

        positive_i=self._dataset.inter_feat[self._dataset.iid_field][valid_indices]
        
        first_occurrence = {}
        for i, uid in enumerate(user_id):
            # 使用 .item() 将 tensor 转换为标量值
            if uid.item() not in first_occurrence:
                first_occurrence[uid.item()] = i

        # 使用 .item() 将每个 uid 转换为普通的数值类型
        positive_u = torch.tensor([first_occurrence[uid.item()] for uid in user_id], dtype=torch.long)

        return [Interaction(data),hist_idx,positive_u,positive_i]#

    def update_config(self, config):
        """
        更新配置，并重新初始化批次大小和步长。
        """
        super().update_config(config)
        self._init_batch_size_and_step()

