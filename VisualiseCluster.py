import argparse
import yaml
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2   
from Database import Database
import numpy as np
import random
import utils
from utils import *
from configuration import Configuration

def main():
    parser = argparse.ArgumentParser(description='Unsupervised Learning of Object Landmarks via Self-Training Correspondence (NeurIPS20)')
    parser.add_argument('--dataset_name', choices=['CelebA','LS3D','Herb'], help='Select training dataset')
    parser.add_argument('--num_workers', default=0, help='Number of workers',type=int)
    parser.add_argument('--experiment_name', help='Name of experiment you from which checkpoint or groundtruth is going to be loaded')

    args=parser.parse_args()

    with open('paths/main.yaml') as file:
        paths = yaml.load(file, Loader=yaml.FullLoader)
    log_path=paths['log_path']

    config=Configuration().params

    utils.initialize_log_dirs(args.experiment_name,log_path)
    path_to_keypoints=utils.get_paths_for_cluster_visualisation(args.experiment_name,log_path)
    keypoints=utils.load_keypoints(path_to_keypoints)
    ShowClusters( keypoints, log_path, args.experiment_name,config.params.M,args.dataset_name)

def ShowVisualRes(keypoints,log_path,experiment_name,number_of_clusters,dataset_name):

    fig = plt.figure(figsize=(34,55))
    gs1 = gridspec.GridSpec(13, 8)
    gs1.update(wspace=0.0, hspace=0.0)
    filenames=[k for k in keypoints.keys() if keypoints[k]['is_it_test_sample']]
    filenames.sort()
    filenames=filenames[:13*8]
    dataset = Database( dataset_name, number_of_clusters,test=True)
    for i in range(len(filenames)):

        ax = plt.subplot(gs1[i])
        plt.axis('off')
        pointstoshow = keypoints[filenames[i]]['prediction']
        image = dataset.getimage_FAN(dataset, filenames[i])
        ax.imshow(image)
        colors = [utils.colorlist[int(i)] for i in np.arange(len(pointstoshow))]
        ax.scatter(pointstoshow[:, 0], pointstoshow[:, 1], s=400, c=colors, marker='P',edgecolors='black', linewidths=0.3)
    fig.show()

    filename = get_logs_path(experiment_name,log_path) / 'Step2.jpg'
    fig.savefig(filename)

    log_text(f"Step2 results created in {filename}", experiment_name,log_path)


def ShowClusters(keypoints,log_path,experiment_name,number_of_clusters,dataset_name):
    dataset = Database( dataset_name, number_of_clusters )

    image_names=list(keypoints.keys())
    random.shuffle(image_names)

    for cluster_number in range(number_of_clusters):
        
        counter_figureimages=0
        counter_datasetimages=0

        fig, subplots= plt.subplots(8,8,figsize=(15,15))
        subplots=subplots.reshape(-1)
        fig.subplots_adjust(wspace=0,hspace=0)

        for s in subplots:
            s.set_axis_off()

        while counter_figureimages<64:

            #for the case where cluster has less than 64 instances
            if(counter_datasetimages>len(keypoints)-1):
                filename = get_logs_path(experiment_name,log_path) / f'Cluster{cluster_number}.jpg'
                fig.savefig(filename)
                break
                
            imagename=image_names[counter_datasetimages]
            imagepoints = keypoints[imagename]

            #if cluster exists in image
            if(sum(imagepoints[:, 2]==cluster_number)>0):
                image = dataset.getimage_FAN(dataset,imagename)
                ax=subplots[counter_figureimages]
                ax.imshow(image)
                ax.scatter(4*imagepoints[imagepoints[:, 2]==cluster_number,0], 4*imagepoints[imagepoints[:, 2]==cluster_number, 1])
                counter_figureimages+=1

            counter_datasetimages+=1

        filename = get_logs_path(experiment_name,log_path) / f'Cluster{cluster_number}.jpg'
        fig.savefig(filename)
        log_text(f"Cluster images created in {filename}", experiment_name,log_path)


if __name__=="__main__":
    main()

