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
from FanClass import FAN_Model
from torch.utils.data import Dataset, DataLoader


def main():

    with open('paths.yml') as file:
        paths = yaml.load(file, Loader=yaml.FullLoader)
    log_path=paths['log_path']
    metadata=paths['metadata']

    config=Configuration().params

    utils.initialize_log_dirs(config.experiment_name,log_path)
    # path_to_keypoints=utils.get_paths_for_cluster_visualisation(config.experiment_name,log_path)
    # keypoints=utils.load_keypoints(path_to_keypoints)

    criterion = nn.MSELoss().cuda()

    FAN = FAN_Model(criterion, 
                    config.experiment_name,
                    config.confidence_thres_FAN, 
                    log_path,
                    1)

    FAN.init_firststage( config.lr,
                        config.weight_decay,
                        config.M,
                        config.bootstrapping_iterations,
                        config.iterations_per_round,
                        config.K,
                        config.nms_thres_FAN,
                        config.lr_step_schedual_stage1)

    cluster_dataset = Database( config.dataset_name, 
                                metadata,
                                function_for_dataloading=Database.get_FAN_inference )
    cluster_dataloader = DataLoader(cluster_dataset, batch_size=config.batchSize, shuffle=False,num_workers=config.num_workers, drop_last=False)
    path_to_checkpoint=config.path_to_checkpoint
    if(path_to_checkpoint is None ):
        path_to_checkpoint=GetPathsResumeFirstStage(config.experiment_name,log_path)
    FAN.load_trained_fiststage_model(path_to_checkpoint)

    keypoints, keypoints_val,_= FAN.Update_pseudoLabels(cluster_dataloader)
    ShowClusters( keypoints_val, log_path, config.experiment_name,config.K, cluster_dataset, config.patch_size)

def ShowClusters(keypoints,log_path,experiment_name,number_of_clusters, dataset, patch_size = -1):
      
    # image_names=list(keypoints.keys())
    # random.shuffle(image_names)

    ###
    image_names=[k for k in keypoints.keys()]
    image_names.sort()
    image_names=image_names[:13*8]
    ###

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
            if(counter_datasetimages>len(keypoints)-1) or counter_datasetimages >= len(image_names):
                filename = get_logs_path(experiment_name,log_path) / f'Cluster{cluster_number}.jpg'
                fig.savefig(filename)
                break
            
            imagename = image_names[counter_datasetimages]
            imagepoints = keypoints[imagename]

            
            image ,_= dataset.Datasource.getimage_FAN(imagename, is_it_test_sample=False)
            ax=subplots[counter_figureimages]

            if(imagepoints.shape[1]==2):
              imagepoints=np.append(imagepoints,np.arange(len(imagepoints)).reshape(-1,1),axis=1)

            # image = np.pad(image, 10)
            x = 4*imagepoints[imagepoints[:, 2]==cluster_number, 0]
            y = 4*imagepoints[imagepoints[:, 2]==cluster_number, 1]
            max_side = image.shape[0]

            if len(x) == 0:
              counter_datasetimages += 1
              continue
            if patch_size != -1:
              image = image[max(0,int(y[0]) - (patch_size // 2)): min(max_side,int(y[0])+(patch_size // 2)), max(0,int(x[0]) - (patch_size // 2)) : min(max_side,int(x[0])+(patch_size // 2)), :]
              ax.imshow(image)
            else:
              ax.imshow(image)
              ax.scatter(x,y)

            counter_figureimages+=1

            counter_datasetimages+=1

        filename = get_logs_path(experiment_name,log_path) / f'Cluster{cluster_number}.jpg'
        fig.savefig(filename)
        log_text(f"Cluster images created in {filename}", experiment_name,log_path)
        plt.close(fig)


if __name__=="__main__":
    main()

