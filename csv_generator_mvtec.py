import os
import glob
import csv
import argparse



def main(args):
    base_path = args.path
    dataset_type=['train','test']
    all_categories = [os.path.join(base_path, o).split('/')[-1] for o in os.listdir(base_path) if os.path.isdir(os.path.join(base_path,o))]
    for dataset in dataset_type:
        for category in all_categories:
            path_category = os.path.join(base_path,category)
            path = os.path.join(path_category,dataset,'**','*')
            if dataset == 'test':
                with open(os.path.join(path_category,category+'_all.csv'), 'w', newline='') as file:
                    print("#### %s csv file %s_all.csv has been generated for %s dataset ####" %('test',category, category.upper()))
                    write_file(file,path)
            else:
                with open(os.path.join(path_category,category+'_train.csv'), 'w', newline='') as file:
                    print("#### %s csv file %s_train.csv has been generated for %s dataset ####" %('train',category,  category.upper()))
                    write_file(file,path)
                    

def write_file(file,path):
    writer = csv.writer(file,delimiter=',', quoting=csv.QUOTE_NONE)
    writer.writerow(['Path','Label'])
    for filename in glob.iglob(path, recursive=True):
        temp=os.path.abspath(filename).split("/")
        if not os.path.isdir(filename):
            if "good" in temp:
                # print(os.path.join(temp[-3],temp[-2],temp[-1])+","+str(1))
                writer.writerow([os.path.join(temp[-3],temp[-2],temp[-1]),str(1)])
            else:
                # print(os.path.join(temp[-3],temp[-2],temp[-1])+","+str(0))
                writer.writerow([os.path.join(temp[-3],temp[-2],temp[-1]),str(0)])

def args_parser():
    parser = argparse.ArgumentParser(description='CSV dataset generator for MVTEC Dataset')

    parser.add_argument('--path', action='store', type=str, required = True, default="",
                        help='path for base MVTEC dataset directory')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=args_parser()
    main(args)