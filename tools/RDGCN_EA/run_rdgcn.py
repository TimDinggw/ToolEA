#!/usr/bin/env python3
import argparse
import sys
import time
import os

from openea.modules.args.args_hander import check_args, load_args
from openea.modules.load.kgs import read_kgs_from_folder_simple
from openea.approaches import RDGCN
from openea.models.basic_model import BasicModel


class ModelFamily(object):
    BasicModel = BasicModel
    RDGCN = RDGCN


def get_model(model_name):
    return getattr(ModelFamily, model_name)


if __name__ == '__main__':
    t = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('args_file', help='Path to the args JSON file')
    parser.add_argument('data_dir', help='Path to the data directory')
    parser.add_argument('--model', default='RDGCN', help='Model name')
    args_cli = parser.parse_args()
    
    args = load_args(args_cli.args_file)
    args.training_data = args_cli.data_dir
    
    print("Embedding module:", args.embedding_module)
    print("Data directory:", args.training_data)
    print("Args:", args)
    
    remove_unlinked = False
    if args.embedding_module == "RSN4EA":
        remove_unlinked = True
    
    kgs = read_kgs_from_folder_simple(
        args.training_data, 
        mode=args.alignment_module, 
        ordered=args.ordered,
        remove_unlinked=remove_unlinked
    )
    
    model = get_model(args.embedding_module)()
    model.set_args(args)
    model.set_kgs(kgs)
    model.init()
    model.run()
    model.test()
    model.save()
    
    print("Total run time = {:.3f} s.".format(time.time() - t))