from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.data.interaction import Interaction
import numpy as np
import torch
from torch.utils.data import BatchSampler,Sampler
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
# from recbole.data.dataloader.abstract_dataloader import (
#     AbstractDataLoader
# )
from recbole.utils import InputType, FeatureType, FeatureSource, ModelType
from recbole.data.transform import construct_transform
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
 
        
    def _change_feat_format(self):
        # 获取经纬度数据，并转换为 torch 张量 
        latitude = torch.tensor(self.item_feat['latitude'].values, dtype=torch.float32)
        longitude = torch.tensor(self.item_feat['longitude'].values, dtype=torch.float32)

        # 如果 GPU 可用，将数据移到 GPU 上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            

    

        self.item_feat['longitude'],self.item_feat['latitude']= self.lat_lon_to_spherical(latitude, longitude,device)
        

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

    def sort(self, by, ascending=True):
        """Sort the interaction records inplace.

        Args:
            by (str or list of str): Field that as the key in the sorting process.
            ascending (bool or list of bool, optional): Results are ascending if ``True``, otherwise descending.
                Defaults to ``True``
        """
        self.inter_feat.sort_values(by=by, ascending=ascending)

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        self._change_feat_format()

        if self.benchmark_filename_list is not None:
            self._drop_unused_col()
            cumsum = list(np.cumsum(self.file_size_list))
            datasets = [
                self.copy(self.inter_feat[start:end])
                for start, end in zip([0] + cumsum[:-1], cumsum)
            ]
            return datasets

        # ordering
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args == "RO":
            self.shuffle()
        elif ordering_args == "TO":
            self.sort(by=self.time_field)
        else:
            raise NotImplementedError(
                f"The ordering_method [{ordering_args}] has not been implemented."
            )

        # splitting & grouping
        split_args = self.config["eval_args"]["split"]
        if split_args is None:
            raise ValueError("The split_args in eval_args should not be None.")
        if not isinstance(split_args, dict):
            raise ValueError(f"The split_args [{split_args}] should be a dict.")

        split_mode = list(split_args.keys())[0]
        assert len(split_args.keys()) == 1
        group_by = self.config["eval_args"]["group_by"]
        if split_mode == "RS":
            if not isinstance(split_args["RS"], list):
                raise ValueError(f'The value of "RS" [{split_args}] should be a list.')
            if group_by is None or group_by.lower() == "none":
                datasets = self.split_by_ratio(split_args["RS"], group_by=None)
            elif group_by == "user":
                datasets = self.split_by_ratio(
                    split_args["RS"], group_by=self.uid_field
                )
            else:
                raise NotImplementedError(
                    f"The grouping method [{group_by}] has not been implemented."
                )
        elif split_mode == "LS":
            datasets = self.leave_one_out(
                group_by=self.uid_field, leave_one_mode=split_args["LS"]
            )
        else:
            raise NotImplementedError(
                f"The splitting_method [{split_mode}] has not been implemented."
            )

        return datasets
    def split_by_ratio(self, ratios, group_by=None):
        """Split interaction records by ratios.

        Args:
            ratios (list): List of split ratios. No need to be normalized.
            group_by (str, optional): Field name that interaction records should grouped by before splitting.
                Defaults to ``None``

        Returns:
            list: List of :class:`~Dataset`, whose interaction features has been split.

        Note:
            Other than the first one, each part is rounded down.
        """
        self.logger.debug(f"split by ratios [{ratios}], group_by=[{group_by}]")
        tot_ratio = sum(ratios)
        ratios = [_ / tot_ratio for _ in ratios]

        if group_by is None:
            tot_cnt = self.__len__()
            split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
            next_index = [
                range(start, end)
                for start, end in zip([0] + split_ids, split_ids + [tot_cnt])
            ]
        else:
            grouped_inter_feat_index = self._grouped_index(
                self.inter_feat[group_by].to_numpy()
            )
            next_index = [[] for _ in range(len(ratios))]
            for grouped_index in grouped_inter_feat_index:
                tot_cnt = len(grouped_index)
                split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
                for index, start, end in zip(
                    next_index, [0] + split_ids, split_ids + [tot_cnt]
                ):
                    index.extend(grouped_index[start:end])
        self._drop_unused_col()
        next_df = [self.inter_feat.iloc[index] for index in next_index]
        next_ds = [self.copy(_) for _ in next_df]
        return next_ds
    
    def __getitem__(self, idx):
        return self.inter_feat.iloc[idx] 





class RandomUserSequenceSampler(Sampler):
    """
    随机用户序列采样器：随机返回每个用户切分后的序列索引
    """
    def __init__(self, dataset, user_field, time_field, seq_len,batch_size):
        self.dataset = dataset
        self.user_field = user_field
        self.time_field = time_field
        self.seq_len = seq_len
        self.splits = self._prepare_splits()
        self.batch_size=batch_size

    def _prepare_splits(self):
        """
        将数据按用户分组，并对时间序列按 seq_len 切分成多个子序列。
        返回一个二维列表：[[seq1_idx], [seq2_idx], ...]
        """
        user_sequences = {}
        # 按用户分组并排序
        for idx, interaction in enumerate(self.dataset):
            user_id = interaction[self.user_field]
            if user_id not in user_sequences:
                user_sequences[user_id] = []
            user_sequences[user_id].append(idx)

        for user_id, indices in user_sequences.items():
            user_sequences[user_id].sort(key=lambda idx: self.dataset[idx][self.time_field])

        # 切分序列
        split_indices = []
        for indices in user_sequences.values():
            for i in range(0, len(indices), self.seq_len):
                split_indices.append(indices[i:i + self.seq_len])
        return split_indices

    def __iter__(self):
        shuffled_indices = np.random.permutation(len(self.splits))
        batches=[]
        for idx in shuffled_indices:
            batches.append(self.splits[idx])
        yield batches

    def __len__(self):
        return len(self.splits)



class AbstractDataLoader(torch.utils.data.DataLoader):
    """:class:`AbstractDataLoader` is an abstract object which would return a batch of data which is loaded by
    :class:`~recbole.data.interaction.Interaction` when it is iterated.
    And it is also the ancestor of all other dataloader.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        _dataset (Dataset): The dataset of this dataloader.
        shuffle (bool): If ``True``, dataloader will shuffle before every epoch.
        pr (int): Pointer of dataloader.
        step (int): The increment of :attr:`pr` for each batch.
        _batch_size (int): The max interaction number for all batch.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.shuffle = shuffle
        self.config = config
        self._dataset = dataset
        self._sampler = sampler
        self._batch_size = self.step = self.model = None
        self._init_batch_size_and_step()
        index_sampler = None
        self.generator = torch.Generator()
        self.generator.manual_seed(config["seed"])
        self.transform = construct_transform(config)
        self.is_sequential = config["MODEL_TYPE"] == ModelType.SEQUENTIAL
        if not config["single_spec"]:
            index_sampler = torch.utils.data.distributed.DistributedSampler(
                list(range(self.sample_size)), shuffle=shuffle, drop_last=False
            )
            self.step = max(1, self.step // config["world_size"])
            shuffle = False

        batch_sampler=RandomUserSequenceSampler(dataset,dataset.uid_field,dataset.time_field,dataset.max_seq_length,self._batch_size)
        super().__init__(
            dataset=list(range(self.sample_size)),
            collate_fn=self.collate_fn,
            num_workers=config["worker"],
            generator=self.generator,
            batch_sampler=batch_sampler
        )

    def _init_batch_size_and_step(self):
        """Initializing :attr:`step` and :attr:`batch_size`."""
        raise NotImplementedError(
            "Method [init_batch_size_and_step] should be implemented"
        )

    def update_config(self, config):
        """Update configure of dataloader, such as :attr:`batch_size`, :attr:`step` etc.

        Args:
            config (Config): The new config of dataloader.
        """
        self.config = config
        self._init_batch_size_and_step()

    def set_batch_size(self, batch_size):
        """Reset the batch_size of the dataloader, but it can't be called when dataloader is being iterated.

        Args:
            batch_size (int): the new batch_size of dataloader.
        """
        self._batch_size = batch_size

    def collate_fn(self):
        """Collect the sampled index, and apply neg_sampling or other methods to get the final data."""
        raise NotImplementedError("Method [collate_fn] must be implemented.")

    def __iter__(self):
        global start_iter
        start_iter = True
        res = super().__iter__()
        start_iter = False
        return res

    def __getattribute__(self, __name: str):
        global start_iter
        if not start_iter and __name == "dataset":
            __name = "_dataset"
        return super().__getattribute__(__name)

class NegSampleDataLoader(AbstractDataLoader):
    """:class:`NegSampleDataLoader` is an abstract class which can sample negative examples by ratio.
    It has two neg-sampling method, the one is 1-by-1 neg-sampling (pair wise),
    and the other is 1-by-multi neg-sampling (point wise).

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=True):
        self.logger = getLogger()
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        

    def _set_neg_sample_args(self, config, dataset, dl_format, neg_sample_args):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.dl_format = dl_format
        self.neg_sample_args = neg_sample_args
        self.times = 1
        if (
            self.neg_sample_args["distribution"] in ["uniform", "popularity"]
            and self.neg_sample_args["sample_num"] != "none"
        ):
            self.neg_sample_num = self.neg_sample_args["sample_num"]

            if self.dl_format == InputType.POINTWISE:
                self.times = 1 + self.neg_sample_num
                self.sampling_func = self._neg_sample_by_point_wise_sampling

                self.label_field = config["LABEL_FIELD"]
                dataset.set_field_property(
                    self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1
                )
            elif self.dl_format == InputType.PAIRWISE:
                self.times = self.neg_sample_num
                self.sampling_func = self._neg_sample_by_pair_wise_sampling
                self.neg_prefix = config["NEG_PREFIX"]
                self.neg_item_id = self.neg_prefix + self.iid_field

                columns = (
                    [self.iid_field]
                    if dataset.item_feat is None
                    else dataset.item_feat.columns
                )
                for item_feat_col in columns:
                    neg_item_feat_col = self.neg_prefix + item_feat_col
                    dataset.copy_field_property(neg_item_feat_col, item_feat_col)
            else:
                raise ValueError(
                    f"`neg sampling by` with dl_format [{self.dl_format}] not been implemented."
                )

        elif (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            raise ValueError(
                f'`neg_sample_args` [{self.neg_sample_args["distribution"]}] is not supported!'
            )

    def _neg_sampling(self, inter_feat):
        if self.neg_sample_args.get("dynamic", False):
            candidate_num = self.neg_sample_args["candidate_num"]
            user_ids = inter_feat[self.uid_field].to_numpy()
            item_ids = inter_feat[self.iid_field].to_numpy()
            neg_candidate_ids = self._sampler.sample_by_user_ids(
                user_ids, item_ids, self.neg_sample_num * candidate_num
            )
            self.model.eval()
            interaction = copy.deepcopy(inter_feat).to(self.model.device)
            interaction = interaction.repeat(self.neg_sample_num * candidate_num)
            neg_item_feat = Interaction(
                {self.iid_field: neg_candidate_ids.to(self.model.device)}
            )
            interaction.update(neg_item_feat)
            scores = self.model.predict(interaction).reshape(candidate_num, -1)
            indices = torch.max(scores, dim=0)[1].detach()
            neg_candidate_ids = neg_candidate_ids.reshape(candidate_num, -1)
            neg_item_ids = neg_candidate_ids[
                indices, [i for i in range(neg_candidate_ids.shape[1])]
            ].view(-1)
            self.model.train()
            return self.sampling_func(inter_feat, neg_item_ids)
        elif (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            user_ids = inter_feat[self.uid_field].to_numpy()
            item_ids = inter_feat[self.iid_field].to_numpy()
            neg_item_ids = self._sampler.sample_by_user_ids(
                user_ids, item_ids, self.neg_sample_num
            )
            return self.sampling_func(inter_feat, neg_item_ids)
        else:
            return inter_feat

    def _neg_sample_by_pair_wise_sampling(self, inter_feat, neg_item_ids):
        inter_feat = inter_feat.repeat(self.times)
        neg_item_feat = Interaction({self.iid_field: neg_item_ids})
        neg_item_feat = self._dataset.join(neg_item_feat)
        neg_item_feat.add_prefix(self.neg_prefix)
        inter_feat.update(neg_item_feat)
        return inter_feat

    def _neg_sample_by_point_wise_sampling(self, inter_feat, neg_item_ids):
        pos_inter_num = len(inter_feat)
        new_data = inter_feat.repeat(self.times)
        new_data[self.iid_field][pos_inter_num:] = neg_item_ids
        new_data = self._dataset.join(new_data)
        labels = torch.zeros(pos_inter_num * self.times)
        labels[:pos_inter_num] = 1.0
        new_data.update(Interaction({self.label_field: labels}))
        return new_data

    def get_model(self, model):
        self.model = model



start_iter=False
class MyTrainDataLoader(NegSampleDataLoader):
    def __init__(self, config, dataset, sampler, shuffle):
        self.sample_size = len(dataset)
        self._batch_size =config["train_batch_size"] 
        super().__init__(config, dataset, sampler, shuffle)
        super()._set_neg_sample_args(config, dataset, InputType.POINTWISE, config["train_neg_sample_args"])
   

    def _init_batch_size_and_step(self):
        self.step = self._batch_size 

    def collate_fn(self, index):
        data=[]
        last=[]
        for idx in index:
            data.append(self._dataset[idx])
            last.append(idx[-1])
        negs=self._neg_sampling(self.dataset[last])
        return data,negs
        


    def _neg_sampling(self, inter_feat):
            if self.neg_sample_args.get("dynamic", False):
                candidate_num = self.neg_sample_args["candidate_num"]
                user_ids = inter_feat[self.uid_field].to_numpy()
                item_ids = inter_feat[self.iid_field].to_numpy()
                neg_candidate_ids = self._sampler.sample_by_user_ids(
                    user_ids, item_ids, self.neg_sample_num * candidate_num
                )
                self.model.eval()
                interaction = copy.deepcopy(inter_feat).to(self.model.device)
                interaction = interaction.repeat(self.neg_sample_num * candidate_num)
                neg_item_feat = Interaction(
                    {self.iid_field: neg_candidate_ids.to(self.model.device)}
                )
                interaction.update(neg_item_feat)
                scores = self.model.predict(interaction).reshape(candidate_num, -1)
                indices = torch.max(scores, dim=0)[1].detach()
                neg_candidate_ids = neg_candidate_ids.reshape(candidate_num, -1)
                neg_item_ids = neg_candidate_ids[
                    indices, [i for i in range(neg_candidate_ids.shape[1])]
                ].view(-1)
                self.model.train()
                return self.sampling_func(inter_feat, neg_item_ids)
            elif (
                self.neg_sample_args["distribution"] != "none"
                and self.neg_sample_args["sample_num"] != "none"
            ):
                user_ids = inter_feat[self.uid_field].to_numpy()
                item_ids = inter_feat[self.iid_field].to_numpy()
                neg_item_ids = self._sampler.sample_by_user_ids(
                    user_ids, item_ids, self.neg_sample_num
                )
                print(neg_item_ids)
                return neg_item_ids#dont mix it
            else:
                return inter_feat
    def __iter__(self):
        batches=[]
        for indices in self.batch_sampler:
            for indice in indices:
                batches.append(indice)
                if len(batches)==self._batch_size:
                    yield self.collate_fn(batches)
                    batches=[]
            if batches:
                yield self.collate_fn(batches)


        
class MyValidDataLoader(NegSampleDataLoader):
    def __init__(self, config, dataset, sampler, shuffle):
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
    def __init__(self, config, dataset, sampler, shuffle):
        # 获取样本大小和初始化参数
        self.sample_size = len(dataset)
        self._sampler = sampler 
        self.batch_size =config["eval_batch_size"] 
        self._init_batch_size_and_step()
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



class AbstractSampler(object):
    """:class:`AbstractSampler` is a abstract class, all sampler should inherit from it. This sampler supports returning
    a certain number of random value_ids according to the input key_id, and it also supports to prohibit
    certain key-value pairs by setting used_ids.

    Args:
        distribution (str): The string of distribution, which is used for subclass.

    Attributes:
        used_ids (numpy.ndarray): The result of :meth:`get_used_ids`.
    """

    def __init__(self, distribution, alpha):
        self.distribution = ""
        self.alpha = alpha
        self.set_distribution(distribution)
        self.used_ids = self.get_used_ids()
 


    def set_distribution(self, distribution):
        """Set the distribution of sampler.

        Args:
            distribution (str): Distribution of the negative items.
        """
        self.distribution = distribution
        if distribution == "popularity":
            self._build_alias_table()

    def _uni_sampling(self, sample_num):
        """Sample [sample_num] items in the uniform distribution.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples.
        """
        raise NotImplementedError("Method [_uni_sampling] should be implemented")

    def _get_candidates_list(self):
        """Get sample candidates list for _pop_sampling()

        Returns:
            candidates_list (list): a list of candidates id.
        """
        raise NotImplementedError("Method [_get_candidates_list] should be implemented")

    def _build_alias_table(self):
        """Build alias table for popularity_biased sampling."""
        candidates_list = self._get_candidates_list()
        self.prob = dict(Counter(candidates_list))
        self.alias = self.prob.copy()
        large_q = []
        small_q = []
        for i in self.prob:
            self.alias[i] = -1
            self.prob[i] = self.prob[i] / len(candidates_list)
            self.prob[i] = pow(self.prob[i], self.alpha)
        normalize_count = sum(self.prob.values())
        for i in self.prob:
            self.prob[i] = self.prob[i] / normalize_count * len(self.prob)
            if self.prob[i] > 1:
                large_q.append(i)
            elif self.prob[i] < 1:
                small_q.append(i)
        while len(large_q) != 0 and len(small_q) != 0:
            l = large_q.pop(0)
            s = small_q.pop(0)
            self.alias[s] = l
            self.prob[l] = self.prob[l] - (1 - self.prob[s])
            if self.prob[l] < 1:
                small_q.append(l)
            elif self.prob[l] > 1:
                large_q.append(l)

    def _pop_sampling(self, sample_num):
        """Sample [sample_num] items in the popularity-biased distribution.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples.
        """

        keys = list(self.prob.keys())
        random_index_list = np.random.randint(0, len(keys), sample_num)
        random_prob_list = np.random.random(sample_num)
        final_random_list = []

        for idx, prob in zip(random_index_list, random_prob_list):
            if self.prob[keys[idx]] > prob:
                final_random_list.append(keys[idx])
            else:
                final_random_list.append(self.alias[keys[idx]])

        return np.array(final_random_list)

    def sampling(self, sample_num):
        """Sampling [sample_num] item_ids.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples and the len is [sample_num].
        """
        if self.distribution == "uniform":
            return self._uni_sampling(sample_num)
        elif self.distribution == "popularity":
            return self._pop_sampling(sample_num)
        else:
            raise NotImplementedError(
                f"The sampling distribution [{self.distribution}] is not implemented."
            )

    def get_used_ids(self):
        """
        Returns:
            numpy.ndarray: Used ids. Index is key_id, and element is a set of value_ids.
        """
        raise NotImplementedError("Method [get_used_ids] should be implemented")

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
                            self.used_ids[key_ids[check_list]],
                            value_ids[check_list],
                        )
                        if v in used
                    ]
                )
        return torch.tensor(value_ids, dtype=torch.long)
class RepeatableSampler(AbstractSampler):
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
        super().__init__(distribution=distribution, alpha=alpha)

    def used_matrix(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        POI_interaction_matrix,category_interaction_matrix,category_ids_counts,unique_IM=counting4all(self.dataset,device)
        itemX = torch.tensor(self.dataset.item_feat["longitude"], dtype=torch.float32)
        itemY = torch.tensor(self.dataset.item_feat["latitude"], dtype=torch.float32)

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
            
        used = [set() for _ in range(result.size(0))]
        for r, c in indices:
            used[r.item()].add(c.item())
        self.used_ids = used

        #print((result.sum())/inter_bool.sum()) 潜在已知感兴趣对象增加量 

        

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.item_num, sample_num)

    def _get_candidates_list(self):
        return list(self.dataset.inter_feat[self.iid_field].to_numpy())

    def get_used_ids(self):
        """
        Returns:
            numpy.ndarray: Used item_ids is the same as positive item_ids.
            Index is user_id, and element is a set of item_ids.
        """
        return np.array([set() for _ in range(self.user_num)])

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
            self.used_matrix()
            return self.sample_by_key_ids(np.arange(len(user_ids)), num)
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
        return torch.tensor(value_ids, dtype=torch.long).reshape(key_num,num)


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
    
