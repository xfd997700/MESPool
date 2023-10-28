# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 17:18:14 2022

@author: Fanding_Xu
"""

import argparse
import torch
from rdkit import RDLogger  
from databuild import TaskConfig, set_seed

from models.MUSE_model import MUSEPred
import rdkit.Chem as Chem
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from IPython.display import display, Image, SVG
from databuild import GenerateData
from databuild import comps_visualize

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Visual Demo", add_help=True)
    
    parser.add_argument('--smiles', type=str,
                        help='smiles of the test molecule')
    parser.add_argument('--name', type=str,
                        help='name of the test molecule')
    
    parser.add_argument('--dataset', type=str, default='bace',
                        help='which benchmark task to run (default: bace)')
    parser.add_argument('--cuda_num', type=int, default=0,
                        help='which gpu to use if any (-1 for cpu, default: 0)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Pytorch DataLoader num_workers (default: 0)')
    
    
    parser.add_argument('--lin_before_conv', action="store_true",
                        help='whether to set a linear layer before convolution layers(default: False)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--hidden_channels_x', type=int, default=128,
                        help='model hidden channels for node embeddings (default: 128)')
    parser.add_argument('--hidden_channels_e', type=int, default=128,
                        help='model hidden channels for edge embeddings (default: 128)')
    parser.add_argument('--threshold', type=int, default=0.5,
                        help='threshold (default: 0.5)')
    args = parser.parse_args()
    
    if args.dataset == 'esol':
        args.threshold = 0.7
    
    
    
    is_cpu = True if args.cuda_num == -1 else False
    set_seed(args.seed, is_cpu)
    RDLogger.DisableLog('rdApp.*')
    task = TaskConfig(args.dataset)
    device = torch.device("cuda:" + str(args.cuda_num)) if not is_cpu and torch.cuda.is_available() else torch.device("cpu")
    
    model = MUSEPred(in_channels_x = 121,
                     in_channels_e = 13,
                     hidden_channels_x = args.hidden_channels_x,
                     hidden_channels_e = args.hidden_channels_e,
                     drop_out = args.dropout,
                     num_classes = task.num_classes,
                     lin_before_conv = args.lin_before_conv,
                     threshold = args.threshold,).to(device)

    model.load_state_dict(torch.load('buffer/' + args.dataset + '.pt'))

    
    mol = Chem.MolFromSmiles(args.smiles)
    name = args.name

    
    data = GenerateData(mol, type_legnth=100).to(device)
    with torch.no_grad():
        model.eval()
        pred = model(data)
    
    if args.dataset == 'esol':
        print("Prediction: {:.4f}\n".format(pred.cpu().item()))
    else:
        print("Prediction: {:.4f}\n".format(pred.sigmoid().cpu().item()))
        
        
    
    multi_fig = True
    imgform = 'svg'
    imgs = comps_visualize(mol, model.comps,
                           model.tars, data.edge_index,
                           size=(800, 600), count_in_node_edges=True,
                           multi_fig = multi_fig, form=imgform)
    
    if imgform == 'png':
        if multi_fig:
            for png in imgs:
                display(Image(png))
        else:
            display(Image(imgs))
    elif imgform == 'svg':
        if multi_fig:
            for idx, svg in enumerate(imgs):
                display(SVG(svg))
                file = 'results/svgs/' + args.dataset + '_' + name + '_' + str(idx) + '.svg'
                with open(file, 'w') as f:
                    f.write(svg)
                    print("Image saved: " + file)
        else:
            display(SVG(imgs))
            file = 'results/svgs/' + args.dataset + '_' + name + '_single.svg'
            with open(file, 'w') as f:
                f.write(svg)
    
   
















