import pickle
import torch
import numpy as np
import random
import os
import argparse
from model import *
import timeit
from data_preprocess import Data_Encoder
from sklearn.model_selection import train_test_split
from torch.utils import data
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy',allow_pickle=True)]


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def init_seed(SEED = 12345):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

def data_pack(batch):
    MAX_PROTEIN_LEN = 545
    MAX_DRUG_LEN = 50

    batch_size = len(batch)

    dfeat_len = batch[0]['atom_feature'].size(1)
    max_atom_length = min([max([it['num_size'] for it in batch]), MAX_DRUG_LEN])

    demb_len = batch[0]['d_v'].size(0)
    pemb_len = batch[0]['p_v'].size(0)

    atoms_new = torch.zeros((batch_size, max_atom_length, dfeat_len), dtype=torch.float)
    adjs_new = torch.zeros((batch_size, max_atom_length, max_atom_length), dtype=torch.float)
    d_new = torch.zeros((batch_size, demb_len), dtype=torch.float)
    p_new = torch.zeros((batch_size, pemb_len), dtype=torch.float)
    dmask_new = torch.zeros((batch_size, demb_len), dtype=torch.long)
    pmask_new = torch.zeros((batch_size, pemb_len), dtype=torch.long)
    num_size_new = torch.zeros((batch_size), dtype=torch.long)
    label_new = torch.zeros((batch_size), dtype=torch.long)

    i = 0
    for it in batch:
        atom_length = min([it['num_size'],max_atom_length])
        atoms_new[i, :atom_length, :] = it['atom_feature'][:atom_length, :]
        adjs_new[i, :atom_length, :atom_length] = it['adj'][:atom_length, :atom_length]
        d_new[i, :] = it['d_v']
        p_new[i, :] = it['p_v']
        dmask_new[i, :] = it['input_mask_d']
        pmask_new[i, :] = it['input_mask_p']
        num_size_new[i] = it['num_size']
        label_new[i] = it['label']
        i = i + 1

    return (atoms_new, adjs_new, num_size_new, d_new, p_new ,dmask_new ,pmask_new ,label_new)


from sklearn.model_selection import StratifiedKFold
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='gpcr')
    parser.add_argument('--model_name', type=str, default='gpcr', help='The name of models')
    parser.add_argument('--atom_dim', type=int, default=75, help='embedding dimension of atoms')
    parser.add_argument('--hid_dim', type=int, default=128, help='embedding dimension of hidden layers')

    parser.add_argument('--dropout', type=float, default=0.2, help='the ratio of Dropout')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--iteration', type=int, default=100, help='the iteration for training')
    parser.add_argument('--n_folds', type=int, default=5, help='the fold count for cross-entropy')
    parser.add_argument('--seed', type=int, default=12345, help='the random seed')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_seed(args.seed)

    params = {'batch_size': args.batch,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True,
              'collate_fn': data_pack}
    Data_set = Data_Encoder("../data/gpcr-train.txt")
    training_set, dev_set = train_test_split(Data_set, test_size=0.2)
    test_set = Data_Encoder("../data/gpcr-test.txt")

    dataset_train = data.DataLoader(training_set, **params)
    dataset_dev = data.DataLoader(dev_set, **params)
    dataset_test = data.DataLoader(test_set , **params)

    model = Predictor(args.hid_dim, device, args.dropout, args.atom_dim, args.batch)
    model.to(device)

    trainer = Trainer(model, args.lr, args.weight_decay, args.batch, len(dataset_train))
    tester = Tester(model)

    if not os.path.exists("./result"):
        os.mkdir("./result")
    if not os.path.exists("./model_end"):
        os.mkdir("./model_end/")
    if not os.path.exists("./result/cos_smi.csv"):
        os.mkdir("./result/cos_smi.csv")
    if not os.path.exists("./result/cos_smi_2_image.csv"):
        os.mkdir("./result/cos_smi_2_image.csv")
    if not os.path.exists("./img"):
        os.mkdir("./img")

    file_AUCs = f'./result/{args.model_name}.txt'
    file_auc_test = f'./result/test_{args.model_name}.txt'
    file_model = f'./model_end/{args.model_name}.pt'

    AUCs = ('best_epoch\t best_AUC_test\t best_AUPR_test\t best_precision_test\tbest_recall_tes')
    with open(file_auc_test, 'w+') as f:
        f.write(AUCs + '\n')
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tPRC_dev\tPrecison_dev\tRecall_dev')
    with open(file_AUCs, 'w+') as f:
        f.write(AUCs + '\n')

    test_auc, test_prc, test_pre, test_recall = 0.0, 0.0, 0.0, 0.0
    best_epoch, best_AUC_test, best_AUPR_test, best_precision_test, best_recall_test = 0.0, 0.0, 0.0, 0.0, 0.0
    max_AUC_dev = 0

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()
    for epoch in range(1, args.iteration+1):
        loss_train = trainer.train(dataset_train, device)
        AUC_dev, PRC_dev, PRE_dev, REC_dev = tester.test(dataset_dev)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time//60, loss_train, AUC_dev,PRC_dev, PRE_dev, REC_dev]
        tester.save_AUCs(AUCs, file_AUCs)
        if AUC_dev > max_AUC_dev:
            tester.save_model(model, file_model)
            max_AUC_dev = AUC_dev

            test_auc, test_prc, test_pre, test_recall = tester.test(dataset_test)
            tester.save_AUCs([epoch, test_auc, test_prc, test_pre, test_recall], file_auc_test)
            best_epoch = epoch
            best_AUC_test = test_auc
            best_AUPR_test = test_prc
            best_precision_test = test_pre
            best_recall_test = test_recall
            print(f'Test ---> AUC: {test_auc}, PRC: {test_prc}')
        print('\t'.join(map(str, AUCs)))

    print('\t'.join(map(str, [best_epoch, best_AUC_test, best_AUPR_test, best_precision_test, best_recall_test])) + '\n')
