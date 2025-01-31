import sys
from logging import getLogger
from recbole.utils import init_logger, init_seed
from Modules.stmamba import Mamba4POI
from Modules.LRU import LRURec
from recbole.model.sequential_recommender import HGN
from recbole.model.sequential_recommender import BERT4Rec
from Modules.TiSASRec import TiSASRec
from recbole.model.sequential_recommender import SRGNN
from recbole.model.sequential_recommender import SHAN
from recbole.model.sequential_recommender import FEARec
from recbole.model.sequential_recommender import SINE
from recbole.model.sequential_recommender import CORE
from recbole.model.sequential_recommender import GRU4Rec
from recbole.model.sequential_recommender import NextItNet
from recbole.config import Config
from utils import *
from recbole.trainer import Trainer
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
import torch
from Modules.myutils import * 

def train(cf,mo):
    config = Config(model=mo, config_file_list=cf)
    dataset = create_dataset(config)
    train_data,valid_data,test_data = data_preparation(config, dataset)

    torch.cuda.empty_cache()
    init_seed(config['seed'], config['reproducibility'])
    
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    logger.info(dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = mo(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = Trainer(config, model)

    best_valid_score, best_valid_result = trainer.fit(
    train_data,
    valid_data,  # 可以保留验证数据集
    verbose=True,    # 保留详细信息，打印结果
    saved=True,      # 根据需要决定是否保存模型参数
    show_progress=False,
    callback_fn=None  # 如果不需要回调函数，可以设置为 None
)

    # model evaluation
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )
    
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    trainer.wandblogger._wandb.finish()
if __name__ == '__main__':
    cfs=[['Configs/Mamba4POI.yaml'],['Configs/FEARec.yaml'],['Configs/LRU4rec.yaml'],['Configs/CORE.yaml'],['Configs/SINE.yaml'],['Configs/TiSASRec.yaml'],['Configs/HGN.yaml'],['Configs/NextItNet.yaml'],['Configs/SRGNN.yaml'],['Configs/BERT4Rec.yaml'],['Configs/SHAN.yaml'],['Configs/GRU4Rec.yaml']]
    mos=[Mamba4POI,FEARec,LRURec,CORE,SINE,TiSASRec,HGN,NextItNet,SRGNN,BERT4Rec,SHAN,GRU4Rec]
    for cf,mo in zip(cfs,mos):
        train(cf,mo)