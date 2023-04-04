import importlib  
import os
import argparse
from sigopt import Connection
import random

ad = importlib.import_module("anomaly-detection")


def get_connection(client_token):
    conn = Connection(client_token=client_token)
    return conn

def create_experiment(conn, experiment_name):
  experiment = conn.experiments().create(
  name=experiment_name,
  parameters=[
    dict(name="optim", categorical_values=["sgd","adam"],type="categorical"),
    dict(name="batch_size", type="int",grid=[32,64,128,256,512]),
    dict(name="lr", type="double",grid=[0.003,0.03,0.3])
  ],
  metrics=[dict(name='loss', objective='minimize'), dict(name='auroc', strategy='store')],)
  return experiment

def evaluate_model(args, assignments):
  print(assignments)  
  args.optim = assignments['optim']
  args.batch_size = assignments['batch_size']
  args.lr = assignments['lr']
  return ad.main(args)

    # d=args.data
    # if args.data:
    #     all_categories = [os.path.join(d, o).split('/')[-1] for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    #     all_categories.sort()
    #     if args.category == 'all':
    #         results=[]
    #         for category in all_categories:
    #             print("\n#### Processing "+category.upper()+ " dataset started ##########\n")
    #             args.category = category
    #             len_inference_data,auroc = ad.main(args)
    #             results.append([category,len_inference_data,auroc])
    #             print("\n#### Processing "+category.upper()+ " dataset completed ########\n")
    #         print(ad.print_datasets_results(results))
    #     else:
    #         # import pdb
    #         # breakpoint()
    #         ad.main(args)


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Anomaly Detection Training and Inference on MVTEC Dataset')

    parser.add_argument('--model', default="resnet50", choices=['resnet18','resnet50'], 
                        help='Backbone architecture for sim-siam and cut-paste feature extractor')

    parser.add_argument('--simsiam', action='store_true', default=False,
                        help='flag to enable simsiam feature extractor')

    parser.add_argument('--cutpaste', action='store_true', default=False,
                        help='flag to enable cut-paste feature extractor')

    parser.add_argument('--image_size', action='store', type=int, default=224,
                        help='image size')

    parser.add_argument('--epochs', action='store', type=int, default=2,
                        help='epochs to train feature extractor')

    parser.add_argument('--batch_size', action='store', type=int, default=64,
                        help='batch size for every forward opeartion')

    parser.add_argument('--optim', action='store', type=str, default='sgd',
                        help='Name of optimizer - sgd/adam')

    parser.add_argument('--data', action='store', type=str, required = True, default="",
                        help='path for base dataset directory')

    parser.add_argument('--category', action='store', type=str, default='hazelnut',
                        help='category of the dataset, i.e. hazelnut')

    parser.add_argument('--freeze_resnet', action='store',  type=int, default=20,
                        help='Epochs upto you want to freeze ResNet layers and only train the new header with FC layers')

    parser.add_argument('--cutpaste_type', default="3way", choices=['normal', 'scar', '3way', 'union'], 
                        help='cutpaste variant to use (dafault: "normal")')

    parser.add_argument('--head_layer', default=2, type=int,
                    help='number of layers in the projection head (default: 2)')
    
    parser.add_argument('--workers', default=56, type=int, help="number of workers to use for data loading (default:56)")

    parser.add_argument('--repeat', default=0, type=int, help="number of test images to use for testing (default:0)")

    parser.add_argument('--dtype', default="fp32", choices=['fp32', 'bf16'], help='data type precision of model inference (dafault: "fp32")')

    parser.add_argument('--model_path', action='store', type=str, default="",
                        help='path for feature extractor model')
    parser.add_argument('--sigopt', action='store_true', default=True,
                        help='Enable if you are using SIGOPT for hyperparameter search')
    parser.add_argument('--ckpt', action='store_true', default=False,
                        help='Enable if you want to save every checkpoint where training loss decreases')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=args_parser()
    print(args)
    conn = get_connection(client_token='XWWRUHJSEEALGLCIZOKNIQBTISNDGTOXJVWETKJSTKFCKHTF')
    experiment = create_experiment(conn,'anomaly_detection')
    print("Created experiment: https://sigopt.com/experiment/" + experiment.id);

    # Run the Optimization Loop between 10x - 20x the number of parameters
    for _ in range(20):
      suggestion = conn.experiments(experiment.id).suggestions().create()
      img_count, auroc, loss = evaluate_model(args,suggestion.assignments)
      print (img_count, loss, auroc)
      conn.experiments(experiment.id).observations().create(
        suggestion=suggestion.id,
        values=[dict(name='loss', value=loss),dict(name='auroc', value=auroc)],
      )
