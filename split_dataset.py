import os
import argparse
import numpy as np
import shutil
import random

def random_split(data, split=0.5):
    random.shuffle(data)
    return data[0:int(len(data)*split)], data[int(len(data)*split):len(data)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--input_folder', required=False, default='input_folder')
    ap.add_argument('--output_folder', required=False, default='output_folder')
    ap.add_argument('--train_split', required=False, default=0.5, type=float)
    args = ap.parse_args()

    print('Dataset:', args.dataset)
    print('Input:', args.input_folder)
    print('Output:', args.output_folder)
    print('Split:', args.train_split)

    input_folder = os.path.join(args.dataset, args.input_folder)
    output_folder = os.path.join(args.dataset, args.output_folder)

    assert os.path.isdir(input_folder), 'Input folder not found'
    assert os.path.isdir(output_folder), 'Output folder not found'

    # print(input_folder)
    # print(output_folder)

    input_data = os.listdir(input_folder)
    output_data = os.listdir(output_folder)

    # print(len(input_data))
    # print(len(output_data))

    assert len(input_data) > 0, 'Input folder is empty'
    assert len(output_data) > 0, 'Output folder is empty'

    trainA, testA = random_split(input_data, split=args.train_split)
    trainB, testB = random_split(output_data, split=args.train_split)

    data = {
        'trainA': trainA, 
        'trainB': trainB,
        'testA': testA,
        'testB': testB
    }

    for folder, files in data.items():   
        src_dir = os.path.join(args.dataset, args.input_folder if('A' in folder) else args.output_folder)
        dst_dir = os.path.join(args.dataset, folder)

        if(not os.path.isdir(dst_dir)):
            os.makedirs(dst_dir)
        
        for f in files:
            shutil.copyfile(os.path.join(src_dir, f), os.path.join(dst_dir, f))