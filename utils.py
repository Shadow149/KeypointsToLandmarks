import torch
import tables
import os
import pickle
import numpy as np
import math
import datetime
import torchvision
import cv2
import glob
from pathlib import Path
import torch.nn as nn
from scipy import linalg
import pandas as pd
from matrix_completion import svt_solve, calc_unobserved_rmse


def GetPathsEval(experiment_name,log_path):
    CheckPointDirectory=get_checkpoints_path(experiment_name,log_path)

    listoffiles=list(str(f.resolve()) for f in CheckPointDirectory.glob('*')) 

    path_to_checkpoint=max([f for f in listoffiles if '.pth' in f and 'SecondStage' in f], key=os.path.getctime)
    log_text('Checkpoint loaded from :'+(str(path_to_checkpoint)),experiment_name,log_path)
    return path_to_checkpoint

def GetPathsResumeFirstStage(experiment_name,log_path):
    CheckPointDirectory=get_checkpoints_path(experiment_name,log_path)

    listoffiles=list(str(f.resolve()) for f in CheckPointDirectory.glob('*')) 

    try:
        path_to_checkpoint=max([f for f in listoffiles if '.pth' in f and 'FirstStage' in f ], key=os.path.getctime)
    except:
        path_to_checkpoint=None
        log_text('Checkpoint was not found',experiment_name,log_path)

    log_text('Checkpoint loaded from :'+(str(path_to_checkpoint)),experiment_name,log_path)
    return path_to_checkpoint

def check_paths(paths):
    assert paths['path_to_superpoint_checkpoint']!=None , "Path missing!! Update 'path_to_superpoint_checkpoint' on paths/main.yaml (link for superpoint_v1.pth availiable on the github repo)"
    assert os.path.isfile(paths['path_to_superpoint_checkpoint']),  f"File {paths['path_to_superpoint_checkpoint']} does not exists. Update 'path_to_superpoint_checkpoint' on paths/main.yaml (link for superpoint_v1.pth availiable on the github repo)" 												


def get_logs_path(experiment_name,log_path):
    log_path=Path(log_path)
    Experiment_Log_directory=log_path / experiment_name / "Logs/"
    return Experiment_Log_directory

def get_checkpoints_path(experiment_name,log_path):
    log_path=Path(log_path)
    CheckPointDirectory=log_path / experiment_name / "CheckPoints/"
    return CheckPointDirectory

def initialize_log_dirs(experiment_name,log_path):

    CheckPointDirectory=get_checkpoints_path(experiment_name,log_path)
    Experiment_Log_directory=get_logs_path(experiment_name,log_path)

    if not Experiment_Log_directory.exists():
        os.makedirs(Experiment_Log_directory)

    if not CheckPointDirectory.exists():
        os.makedirs(CheckPointDirectory)

def log_text(text,experiment_name,log_path):

    Experiment_Log_directory=get_logs_path(experiment_name,log_path)
    Log_File=Experiment_Log_directory / (experiment_name + '.txt')

    print(text + "  (" + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + ")")

    f = open(Log_File, 'a')
    f.write(text + "  (" + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + ")" + '\n')
    f.close()


def load_keypoints(filename):
    checkPointFile=filename
    with open(checkPointFile, 'rb') as handle:
        keypoints=pickle.load( handle)
    return keypoints


def save_keypoints(Image_Keypoints,filename):
    with open(filename, 'wb') as handle:
        pickle.dump(Image_Keypoints, handle, protocol=pickle.HIGHEST_PROTOCOL)



def my_cuda(model):
    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model



def CreateFileArray(name,columns):
    filename = name+f'.npy'
    if os.path.exists(filename):
        os.remove(filename)
    f = tables.open_file(filename, mode='w')
    atom = tables.Float64Atom()
    f.create_earray(f.root, 'data', atom, (0, columns))
    f.close()

def AppendFileArray(array,name):
    filename = name+f'.npy'
    f = tables.open_file(filename, mode='a')
    f.root.data.append(array)
    f.close()


def OpenreadFileArray(name):
    filename = name+f'.npy'
    f = tables.open_file(filename, mode='r')
    a=f.root.data
    return a,f

def ClosereadFileArray(f,name):
    filename = name+f'.npy'
    f.close()
    if os.path.exists(filename):
        os.remove(filename)



def BuildGaussians(keypoints,resolution=64,size=1):
    points = keypoints.copy()
    points[:, 0] = points[:, 0] + 1
    points[:, 1] = points[:, 1] + 1


    numberOfAnnotationsPoints = points.shape[0]
    if (numberOfAnnotationsPoints == 0):
        heatMaps=torch.zeros(1, resolution, resolution)
    else:
        heatMaps = torch.zeros(numberOfAnnotationsPoints, resolution, resolution)
    for i in range(numberOfAnnotationsPoints):
        p=np.asarray(points[i])
        try:
            heatMaps[i] = fastDrawGaussian(heatMaps[i], p, size)

        except:
            pass
    heatmap = torch.max(heatMaps, 0)[0]
    return heatmap

def BuildMultiChannelGaussians(outputChannels,keypoints,resolution=64,size=3):
    points = keypoints.copy()
    points[:, 0] = points[:, 0] + 1
    points[:, 1] = points[:, 1] + 1

    numberOfAnnotationsPoints = points.shape[0]
    heatMaps=torch.zeros(outputChannels, resolution, resolution)

    for i in range(numberOfAnnotationsPoints):
        p=np.asarray(points[i]) 
        try:
            heatMaps[int(p[2])] = fastDrawGaussian(heatMaps[int(p[2])], p, size)                                                                                                    
        except:
            pass

    return heatMaps



def GetBatchMultipleHeatmap(confidenceMap,threshold,NMSthes=2):

    mask=confidenceMap>threshold
    prob =confidenceMap[mask]
    pred=torch.nonzero(mask)
    points = pred[:, 2:4]
    points=points.flip(1)
    idx =pred[:, 0]
    nmsPoints=torch.cat((points.float(),prob.unsqueeze(1)),1).T
    newpoints = torch.cat((nmsPoints[0:1, :] - NMSthes, nmsPoints[1:2, :] - NMSthes, nmsPoints[0:1, :] + NMSthes,
                           nmsPoints[1:2, :] + NMSthes, nmsPoints[2:3, :]), 0).T

    res = torchvision.ops.boxes.batched_nms(newpoints[:, 0:4], newpoints[:, 4],idx, 0.01)
    p=torch.cat((pred[res,:1].float(),nmsPoints[:,res].T,pred[res,1:2].float()),dim=1)
    value, indices = p[:,0].sort()
    p=p[indices]
    return p


def GetDescriptors(descriptor_volume, points, W, H):
    D = descriptor_volume.shape[0]
    if points.shape[0] == 0:
        descriptors = torch.zeros((0, D))
    else:
        coarse_desc = descriptor_volume.unsqueeze(0)
        samp_pts = points.clone().T
        samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
        samp_pts = samp_pts.transpose(0, 1).contiguous()
        samp_pts = samp_pts.view(1, 1, -1, 2)
        samp_pts = samp_pts.float()
        densedesc = torch.nn.functional.grid_sample(coarse_desc, samp_pts,align_corners=False)
        densedesc = densedesc.view(D, -1)
        densedesc /= torch.norm(densedesc, dim=0).unsqueeze(0)
        descriptors = densedesc.T
    return descriptors


def GetPointsFromHeatmaps(heatmapOutput):
    # get max for each batch sample
    keypoints = torch.zeros(heatmapOutput.size(0), 4)

    val, idx = torch.max(heatmapOutput.view(heatmapOutput.shape[0], -1), 1)
    keypoints[:, 2] = val
    keypoints[:, :2] = idx.view(idx.size(0), 1).repeat(1, 1, 2).float()
    keypoints[..., 0] = (keypoints[..., 0] - 1) % heatmapOutput.size(2) + 1
    keypoints[..., 1] = keypoints[..., 1].add_(-1).div_(heatmapOutput.size(1)).floor()
    keypoints[:, 3] = torch.arange(heatmapOutput.size(0))

    keypoints[:, :2] = 4 * keypoints[:, :2]
    return keypoints


def fastDrawGaussian(img,pt,size):
    if (size == 3):
        g = gaussian3
    elif (size == 1):
        g = gaussian1
    s = 1
    ul = torch.tensor([[math.floor(pt[0] - s)], [math.floor(pt[1] -s)]])
    br = torch.tensor([[math.floor(pt[0] + s)], [math.floor(pt[1] +s)]])
    if (ul[0] > img.shape[1] or ul[1] > img.shape[0] or br[0] < 1 or br[1] < 1):
        return img

    g_x = torch.tensor([[max(1, -ul[0])], [min(br[0], img.shape[1]) - max(1, ul[0]) + max(1, -ul[0])]])
    g_y = torch.tensor([[max(1, -ul[1])], [min(br[1], img.shape[0]) - max(1, ul[1]) + max(1, -ul[1])]])
    img_x = torch.tensor([[max(1, ul[0])], [min(br[0], img.shape[1])]])
    img_y = torch.tensor([[max(1, ul[1])], [min(br[1], img.shape[0])]])

    assert (g_x[0] > 0 and g_y[0] > 0)
    img[int(img_y[0])-1:int(img_y[1]), int(img_x[0])-1:int(img_x[1])] += g[int(g_y[0])-1:int(g_y[1]), int(g_x[0])-1:int(g_x[1])]
    return img


def fwd_withNan_calculation(keypoints_array,groundtruth_array,is_test_sample,reg_factor=0.1,npts=10,nimages=[300,19000],nrepeats=10,size=218,compute_iod=None):
    
    Xtr=keypoints_array[is_test_sample==0].copy()
    Xtest=keypoints_array[is_test_sample==1].copy()

    Xtr_new = Xtr.copy()
    Xtest_new = Xtest.copy()

    nl = int(Xtr.shape[1])
    DF = pd.DataFrame(Xtr_new)
    col_means = DF.apply(np.mean, 0)
    Xc_tr_mean = DF.fillna(value=col_means).to_numpy()/size
    Xc_tr = Xc_tr_mean.copy()
    mask = np.ones_like(Xtr_new.reshape(len(Xtr_new),nl))
    mask[np.where(np.isnan(Xtr_new.reshape(len(Xtr_new),nl)))] = 0

    R_hat = svt_solve(Xc_tr, np.round(mask))
    Xc_tr = size * R_hat

    Xc_tr[np.where(mask==1)] = Xtr_new.reshape(len(Xtr_new),nl)[np.where(mask==1)]

    Xtr_new=Xc_tr
    
    DF = pd.DataFrame(Xtest_new.reshape(Xtest_new.shape[0],nl))
    Xtest_new = DF.fillna(value=col_means).to_numpy()

    
    Ytr=groundtruth_array[is_test_sample==0].copy()
    Ytest=groundtruth_array[is_test_sample==1].copy()

    allres,allres_perlandmark=compute_errors(Xtr_new,Ytr,Xtest_new,Ytest,reg_factor=reg_factor,npts=npts,nimages=nimages,nrepeats=nrepeats,size=size,fwd=True,compute_iod=compute_iod)
    fwd=np.average(allres,axis=1)
    fwd_perlandmark=np.average(allres_perlandmark,axis=2)[0]
    fwd_perlandmark.sort()
    fwd_perlandmark_cumulative = np.cumsum(fwd_perlandmark)
    fwd_perlandmark_cumulative = fwd_perlandmark_cumulative / np.arange(1, len(fwd_perlandmark_cumulative) + 1)

    return fwd,fwd_perlandmark_cumulative

def bwd_withNan_calculation(keypoints_array,groundtruth_array,is_test_sample,reg_factor=0.1,npts=10,nimages=[300,19000],nrepeats=10,size=218,compute_iod=None):
    
    Xtr=keypoints_array[is_test_sample==0].copy()
    Ytr=groundtruth_array[is_test_sample==0].copy()
    Xtest=keypoints_array[is_test_sample==1].copy()
    Ytest=groundtruth_array[is_test_sample==1].copy()


    Xtr=Xtr.reshape(len(Xtr),-1,2)
    Xtest=Xtest.reshape(len(Xtest),-1,2)

    number_of_landmarks=npts
    backward_per_landmark=np.zeros((len(nimages),number_of_landmarks,))
    for j in range(number_of_landmarks):

        Xtr_landmark=Xtr[:,j]
        landmarknotnan=(~np.isnan(Xtr_landmark))[:, 0]
        Xtr_landmark=Xtr_landmark[landmarknotnan]
        Ytr_landmark=Ytr[landmarknotnan]

        Xtest_landmark=Xtest[:,j]
        landmarknotnan=(~np.isnan(Xtest_landmark))[:, 0]
        Xtest_landmark=Xtest_landmark[landmarknotnan]
        Ytest_landmark=Ytest[landmarknotnan]

        allres,allres_perlandmark=compute_errors(Ytr_landmark,Xtr_landmark,Ytest_landmark,Xtest_landmark,reg_factor=reg_factor,npts=1,nimages=nimages,nrepeats=nrepeats,size=size,fwd=False,compute_iod=compute_iod)
        backward_per_landmark[:,j]=np.average(allres,axis=1)
    

    bwd=np.average(backward_per_landmark,axis=1)
    bwd_perlandmark=backward_per_landmark
    bwd_perlandmark.sort()
    bwd_perlandmark_cumulative = np.cumsum(bwd_perlandmark)
    bwd_perlandmark_cumulative = bwd_perlandmark_cumulative / np.arange(1, len(bwd_perlandmark_cumulative) + 1)

    return bwd,bwd_perlandmark_cumulative


def fwd_calculation(keypoints_array,groundtruth_array,is_test_sample,reg_factor,npts,nimages,nrepeats,size,compute_iod):
    Xtr=keypoints_array[is_test_sample==0].copy()
    Ytr=groundtruth_array[is_test_sample==0].copy()
    Xtest=keypoints_array[is_test_sample==1].copy()
    Ytest=groundtruth_array[is_test_sample==1].copy()

    allres,allres_perlandmark=compute_errors(Xtr,Ytr,Xtest,Ytest,reg_factor=reg_factor,npts=npts,nimages=nimages,nrepeats=nrepeats,size=size,fwd=True,compute_iod=compute_iod)
    fwd=np.average(allres,axis=1)
    fwd_perlandmark=np.average(allres_perlandmark,axis=2)[0]
    fwd_perlandmark.sort()
    fwd_perlandmark_cumulative = np.cumsum(fwd_perlandmark)
    fwd_perlandmark_cumulative = fwd_perlandmark_cumulative / np.arange(1, len(fwd_perlandmark_cumulative) + 1)

    return fwd,fwd_perlandmark_cumulative


def regressed_calculation(keypoints_array,groundtruth_array,is_test_sample,reg_factor,nimages,size):
    Xtr=keypoints_array[is_test_sample==0].copy()
    Ytr=groundtruth_array[is_test_sample==0].copy()
    Xtest=keypoints_array[is_test_sample==1].copy()
    Ytest=groundtruth_array[is_test_sample==1].copy()

    X_regressed,X_gt=getRegressed(Xtr,Ytr,Xtest,Ytest,reg_factor=reg_factor,nimages=nimages,size=size)


    return X_regressed,X_gt



def bwd_calculation(keypoints_array,groundtruth_array,is_test_sample,reg_factor,npts,nimages,nrepeats,size,compute_iod):
    Xtr=keypoints_array[is_test_sample==0].copy()
    Ytr=groundtruth_array[is_test_sample==0].copy()
    Xtest=keypoints_array[is_test_sample==1].copy()
    Ytest=groundtruth_array[is_test_sample==1].copy()

    allres,allres_perlandmark=compute_errors(Ytr,Xtr,Ytest,Xtest,reg_factor=reg_factor,npts=npts,nimages=nimages,nrepeats=nrepeats,size=size,fwd=False,compute_iod=compute_iod)
    bwd=np.average(allres,axis=1)
    bwd_perlandmark=np.average(allres_perlandmark,axis=2)[0]
    bwd_perlandmark.sort()
    bwd_perlandmark_cumulative = np.cumsum(bwd_perlandmark)
    bwd_perlandmark_cumulative = bwd_perlandmark_cumulative / np.arange(1, len(bwd_perlandmark_cumulative) + 1)

    return bwd,bwd_perlandmark_cumulative


def compute_errors(Xtr,Ytr,Xtest,Ytest,reg_factor,npts,nimages,nrepeats,size,compute_iod,fwd=True):
    n = nimages
    all_errors = np.zeros((len(n),nrepeats))
    all_perlandmarkerrors=np.zeros((len(n),int(Ytest.shape[1]/2),nrepeats))
    for tmp_idx in range(0,len(n)):
        for j in range(0,nrepeats):
            idx = np.random.permutation((range(0,Xtr.shape[0])))[0:n[tmp_idx]+1]
            R, X0, Y0 = train_regressor(Xtr[idx,:], Ytr[idx,:], reg_factor,size,'type2')
            err = np.zeros((Xtest.shape[0]))
            err_perlandmark = np.zeros((Xtest.shape[0],int(Ytest.shape[1]/2)))
            for i in range(0,Xtest.shape[0]):
                x = Xtest[i,:]
                y = Ytest[i,:]
                if fwd:
                    x = fit_regressor(R,x,X0,Y0,size,'type2')
                    iod = compute_iod(y.reshape(-1,2))
                    distances = np.sqrt(np.sum((y.reshape(-1,2) - x)**2, axis=-1))
                    err[i]=mean_error = np.mean(distances / iod)
                    err_perlandmark[i]=perlandmark=distances /iod
                else:
                    iod = compute_iod(x.reshape(-1,2))
                    x = fit_regressor(R,x,X0,Y0,size,'type2')
                    y = y.reshape(-1,2)
                    
                    err_perlandmark[i]=np.sqrt(np.sum((x-y)**2,1))/iod
                    err[i] = np.sum(np.sqrt(np.sum((x-y)**2,1)))/(iod*npts)
            all_errors[tmp_idx,j] = np.mean(err)
            all_perlandmarkerrors[tmp_idx,:,j]=np.mean(err_perlandmark,axis=0)
    return all_errors  ,all_perlandmarkerrors      
    

def getRegressed(Xtr,Ytr,Xtest,Ytest,reg_factor,nimages,size):
    n = nimages


    X_regressed=np.zeros_like(Ytest)

    for tmp_idx in range(0,len(n)):

        idx = np.random.permutation((range(0,Xtr.shape[0])))[0:n[tmp_idx]+1]
        R, X0, Y0 = train_regressor(Xtr[idx,:], Ytr[idx,:], reg_factor,size,'type2')
        for i in range(0,Xtest.shape[0]):
            x = Xtest[i,:]
            y = Ytest[i,:]

            x = fit_regressor(R,x,X0,Y0,size,'type2')
            X_regressed[i]=x.reshape(-1)

    return X_regressed,Ytest



def train_regressor(X,Y,l,center=128.0,option=None):
    if option == 'type0':
        C = X.transpose() @ X
        R = ( Y.transpose() @ X ) @ linalg.inv( C + l*(C.max()+1e-12)*np.eye(X.shape[1]))
        X0 = 1.0
        Y0 = 1.0
    elif option == 'type1':
        Xtmp = X/center - 0.5
        C = Xtmp.transpose() @ Xtmp
        Ytmp = Y/center - 0.5
        R = ( Ytmp.transpose() @ Xtmp ) @ linalg.inv( C + l*(C.max()+1e-12)*np.eye(Xtmp.shape[1]))
        X0 = 1.0
        Y0 = 1.0
    elif option == 'type2':
        Xtmp = X/center - 0.5
        X0 = Xtmp.mean(axis=0, keepdims=True)
        Xtmp = Xtmp - np.ones((Xtmp.shape[0],1)) @ X0.reshape(1,-1)
        C = Xtmp.transpose() @ Xtmp
        Ytmp = Y/center - 0.5
        Y0 = Ytmp.mean(axis=0, keepdims=True)
        Ytmp = Ytmp - np.ones((Ytmp.shape[0],1)) @ Y0.reshape(1,-1)
        R = ( Ytmp.transpose() @ Xtmp ) @ linalg.inv( C + l*(C.max()+1e-12)*np.eye(Xtmp.shape[1])) 
    elif option == 'type3':
        Xtmp = X
        X0 = Xtmp.mean(axis=0, keepdims=True)
        Xtmp = Xtmp - np.ones((Xtmp.shape[0],1)) @ X0.reshape(1,-1)
        C = Xtmp.transpose() @ Xtmp
        Ytmp = Y
        Y0 = Ytmp.mean(axis=0, keepdims=True)
        Ytmp = Ytmp - np.ones((Ytmp.shape[0],1)) @ Y0.reshape(1,-1)
        R = ( Ytmp.transpose() @ Xtmp ) @ linalg.inv( C + l*(C.max()+1e-12)*np.eye(Xtmp.shape[1]))
    return R, X0, Y0


def fit_regressor(R,x,X0,Y0,center=128.0,option=None):
    if option == 'type0':
        x = (R @ x).reshape(-1,2)
    elif option == 'type1':
        x = (R @ (x/center - 0.5).transpose()).reshape(-1,2)
        x = (x + 0.5)*center
    elif option == 'type2':
        x = (R @ (x/center - 0.5 - X0).transpose()).reshape(-1,2) + Y0.reshape(-1,2)
        x = (x + 0.5)*center
    elif option == 'type3':
        x = (R @ (x - X0).transpose()).reshape(-1,2) + Y0.reshape(-1,2)
    return x









def gaussian(size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
             height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5, mean_vert=0.5):
        # handle some defaults
        if width is None:
            width = size
        if height is None:
            height = size
        if sigma_horz is None:
            sigma_horz = sigma
        if sigma_vert is None:
            sigma_vert = sigma
        center_x = mean_horz * width + 0.5
        center_y = mean_vert * height + 0.5
        gauss = np.empty((height, width), dtype=np.float32)
        # generate kernel
        for i in range(height):
            for j in range(width):
                gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (sigma_horz * width), 2) / 2.0 + math.pow(
                    (i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
        if normalize:
            gauss = gauss / np.sum(gauss)
        return gauss


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx].unsqueeze(1)),
                    heatmap_gt.mul(target_weight[:, idx].unsqueeze(1))
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints 





def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout



gaussian3=torch.tensor([[0.16901332, 0.41111228, 0.16901332],
       [0.41111228, 1.        , 0.41111228],
       [0.16901332, 0.41111228, 0.16901332]])


gaussian1=torch.tensor([[0.0, 0.0, 0.0],
       [0.0, 1.  , 0.0],
       [0.0, 0.0, 0.0]])



colorlist = [
	"#ffdd41",
	"#0043db",
	"#62ef00",
	"#ff34ff",
	"#00ff5e",
	"#ef00de",
	"#00bd00",
	"#8f00c3",
	"#e5f700",
	"#a956ff",
	"#4bba00",
	"#ee00c2",
	"#4cbb0e",
	"#ff00c2",
	"#00ffa8",
	"#fe60f9",
	"#55b200",
	"#0052e5",
	"#ffe000",
	"#001e96",
	"#f1e215",
	"#336dff",
	"#e9d800",
	"#6056e7",
	"#ffd910",
	"#0070ff",
	"#8cbb00",
	"#0041c2",
	"#61bf2c",
	"#2a007b",
	"#00b64b",
	"#8237c0",
	"#00c87c",
	"#750091",
	"#00ffd3",
	"#f50000",
	"#00ffff",
	"#ff003a",
	"#00ffff",
	"#d40000",
	"#00ffff",
	"#c70000",
	"#00ffff",
	"#cb0008",
	"#00ffff",
	"#ed4000",
	"#00ffff",
	"#ff005d",
	"#00e8c5",
	"#d20022",
	"#00ffff",
	"#b00000",
	"#00f2f6",
	"#d50031",
	"#00edd2",
	"#ce004e",
	"#00c47f",
	"#7f56e3",
	"#ffa900",
	"#0046c7",
	"#d0b325",
	"#001175",
	"#ff8600",
	"#0080fb",
	"#ca9700",
	"#0060d9",
	"#ac9a00",
	"#006de2",
	"#f2a533",
	"#0088ff",
	"#d74900",
	"#0066d8",
	"#618a00",
	"#d679ff",
	"#077200",
	"#ff88ff",
	"#008a2d",
	"#590077",
	"#00d19a",
	"#c8005b",
	"#00f0e1",
	"#ac000f",
	"#00f0ff",
	"#a10000",
	"#00ecff",
	"#c12d0c",
	"#00e9ff",
	"#d34f00",
	"#005fd0",
	"#6a8600",
	"#0060cd",
	"#c67900",
	"#0066d1",
	"#9b8700",
	"#210052",
	"#ffe585",
	"#1a0045",
	"#00b87b",
	"#d665cf",
	"#005600",
	"#ff97ff",
	"#005100",
	"#ff9dff",
	"#005100",
	"#e685e5",
	"#1a5800",
	"#ffa6ff",
	"#004900",
	"#ff6eb9",
	"#00540e",
	"#d49dff",
	"#004500",
	"#ffb1ff",
	"#004300",
	"#dd337b",
	"#00e3d0",
	"#940000",
	"#00e8ff",
	"#ae191d",
	"#00e3f8",
	"#860000",
	"#70faff",
	"#860000",
	"#69f6fa",
	"#790000",
	"#00d6ff",
	"#a21221",
	"#00d6e0",
	"#bf1b3d",
	"#00d5dd",
	"#780002",
	"#00d3ff",
	"#730000",
	"#00d2ff",
	"#ac4f00",
	"#00a6ff",
	"#f37e3b",
	"#0081e3",
	"#496000",
	"#533d9a",
	"#138b53",
	"#7c005e",
	"#00814b",
	"#60005b",
	"#005919",
	"#ffb7ff",
	"#004000",
	"#ffb9ff",
	"#003800",
	"#cca8ff",
	"#003200",
	"#ff7bab",
	"#003200",
	"#ff7ca1",
	"#002f00",
	"#ff84a8",
	"#002900",
	"#cebbff",
	"#324d00",
	"#0094f1",
	"#765400",
	"#00aaff",
	"#693600",
	"#00adff",
	"#6b4400",
	"#0094eb",
	"#595900",
	"#006bc4",
	"#dbb670",
	"#001f6a",
	"#ffc49b",
	"#000432",
	"#eef6e3",
	"#230028",
	"#bef9ff",
	"#a90041",
	"#00bfc1",
	"#85002e",
	"#63dbe9",
	"#750024",
	"#00c7ff",
	"#480900",
	"#21c6ff",
	"#4c1400",
	"#00bdff",
	"#655400",
	"#0096e6",
	"#ffa078",
	"#001d57",
	"#ffac96",
	"#382b7f",
	"#374900",
	"#0091e2",
	"#3f4900",
	"#00bdff",
	"#360000",
	"#6adcff",
	"#350000",
	"#61d9ff",
	"#6c002f",
	"#00c4f8",
	"#813529",
	"#7cdbfd",
	"#1e0a00",
	"#ede4ff",
	"#001400",
	"#ece4ff",
	"#001b00",
	"#ffd9ef",
	"#002900",
	"#e1e1ff",
	"#002500",
	"#aeadee",
	"#002400",
	"#eaa6a5",
	"#002200",
	"#b0709f",
	"#004822",
	"#003a7d",
	"#a4965a",
	"#004b86",
	"#8c7d42",
	"#005b99",
	"#483000",
	"#008cc3",
	"#9a633e",
	"#005083",
	"#a68c61",
	"#001e3d",
	"#82764a",
	"#003054",
	"#1c3100",
	"#4d2c57",
	"#009b9e",
	"#2b1500",
	"#00929e",
	"#441c30",
	"#127c72",
	"#261700",
	"#005b88",
	"#2a2800",
	"#00456f",
	"#003c19",
	"#7f6788",
	"#062300",
	"#505880",
	"#004c29",
	"#00496e",
	"#2b2500",
	"#628687",
	"#020d14",
	"#005e6b",
	"#271b00",
	"#00546d",
	"#1d1617",
	"#264133",
	"#252737",
	"#002f24",
	"#002f3d",
	"#001919"
]