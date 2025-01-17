import argparse
import types

class Configuration():

    def __init__(self):
        parser = argparse.ArgumentParser(description='Unsupervised Learning of Object Landmarks via Self-Training Correspondence')
        

        parser.add_argument('--experiment_name',help='Please assign a unique name for each experiment. Use the same name for both training set 1 and 2.',required=True)
        parser.add_argument('--dataset_name', choices=['CelebA','LS3D','Herb'], default='CelebA',help='Select training dataset')
        parser.add_argument('--K', default=10 ,help='Select number of discovered landmarks K')
        parser.add_argument('--gpunum', default=1)
        parser.add_argument('--num_workers', default=0, help='Number of workers',type=int)
        parser.add_argument('--resume', action='store_true', help='If True stage 1 and 2 will resume form last saved checkpoint and pseudogroundtruth.')
        parser.add_argument('--stage',default=1,help='Speficy the stage of the algorithm you want to evaluate on eval.py')
        parser.add_argument('--path_to_checkpoint',default=None)
        parser.add_argument('--patch_size',default=-1)


        args = parser.parse_args()


        hyperparameters=types.SimpleNamespace()

        hyperparameters.experiment_name=args.experiment_name
        hyperparameters.dataset_name=args.dataset_name
        hyperparameters.gpunum=int(args.gpunum)
        hyperparameters.num_workers=args.num_workers
        hyperparameters.resume=args.resume
        hyperparameters.path_to_checkpoint=args.path_to_checkpoint
        hyperparameters.K=int(args.K)
        hyperparameters.eval_Stage=int(args.stage)

        #params Whole Pipeline
        hyperparameters.lr=1e-4
        hyperparameters.weight_decay=1e-5
        hyperparameters.batchSize=16
        hyperparameters.useflip=True

        #params Stage 1        
        hyperparameters.batchSize_superpoint=16
        hyperparameters.confidence_thres_superpoint=0.25
        hyperparameters.bootstrapping_iterations=15000 # First training before keypoints found...?
        hyperparameters.iterations_per_round=2500 # How often clusters are done
        hyperparameters.total_iterations_stage1=100000 # Max iters
        hyperparameters.remove_superpoint_outliers_percentage=0.5
        hyperparameters.M=100
        hyperparameters.confidence_thres_FAN=0.15
        hyperparameters.nms_thres_FAN=2
        hyperparameters.lr_step_schedual_stage1=[75000,90000]


        hyperparameters.nms_thres_superpoint=8
        

        #params Stage 2
        hyperparameters.lr_step_schedual_stage2=[50000,70000]      
        hyperparameters.total_iterations_stage2=200000
        hyperparameters.save_checkpoint_frequency=20000


        #scale for different number of gpus
        hyperparameters.iterations_per_round=int(hyperparameters.iterations_per_round/hyperparameters.gpunum)
        hyperparameters.lr=hyperparameters.lr*hyperparameters.gpunum
        hyperparameters.batchSize=hyperparameters.batchSize*hyperparameters.gpunum

        hyperparameters.patch_size=int(args.patch_size)

        self.params=hyperparameters



