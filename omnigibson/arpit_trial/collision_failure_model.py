
import argparse
import numpy as np
import os
import torch
import importlib
from pointnet2.models import action_pointnet2_cls_ssg
from pointnet2.data_utils.utils import generate_point_cloud_from_depth


class CollisionFailureModel:
    def __init__(self):  
        '''MODEL LOADING'''
        num_class = 1
        experiment_dir = "/home/arpit/test_projects/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_cls_ssg_wo_floors_1000_corrected"
        # model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
        # model_type = "action_pointnet2_cls_ssg"
        # model = importlib.import_module(model_type)

        self.classifier = action_pointnet2_cls_ssg.get_model(num_class, normal_channel=False)
        self.classifier = self.classifier.cuda()

        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()

        # with torch.no_grad():
        #     instance_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        #     print('Test Instance Accuracy: ', instance_acc)

    def get_pcd(self, obs, obs_info, robot_name):
        depth = obs[robot_name][f"{robot_name}:eyes:Camera:0"]["depth"].numpy()
        intr =  np.array([
            [103.8416,   0.0000,  64.0000],
            [  0.0000, 103.8416,  64.0000],
            [  0.0000,   0.0000,   1.0000]])
        
        # creating mask to remove floors
        seg_semantic = obs[robot_name][f"{robot_name}:eyes:Camera:0"]["seg_semantic"].cpu().numpy()
        seg_semantic_info = obs_info[robot_name][f"{robot_name}:eyes:Camera:0"]["seg_semantic"]

        pcd_points = []
        pcd_normals = []
        pcd_colors = []

        # creating mask to remove floors
        floor_id = -1
        for k, v in seg_semantic_info.items():
            sem_id, class_name = k, v
            if class_name == 'floors':
                floor_id = sem_id
                break

        if floor_id != -1:
            mask = np.zeros_like(depth)
            mask[seg_semantic != floor_id] = 1
        else:
            mask = np.ones_like(depth)

        o3d_pcd = generate_point_cloud_from_depth(depth, intr, mask)
        pcd_points.append(np.asarray(o3d_pcd.points))
        pcd_colors.append(np.asarray(o3d_pcd.colors))
        pcd_normals.append(np.asarray(o3d_pcd.normals))
        
        # TODO: Convert to torch tensors of dtype float64
        # pcd shape: torch.Size([1, 16384, 3])
        # actions shape: torch.Size([1, 10])

        pcd = dict()
        pcd['points'] = pcd_points
        pcd['colors'] = pcd_colors
        pcd['normals'] = pcd_normals
        return pcd
    
    def check_collision(self, obs, obs_info, actions, robot_name):
        mean_correct = []
        collision_count, false_negative, false_positive = 0, 0, 0  
        pcd = self.get_pcd(obs, obs_info, robot_name)
        points = torch.from_numpy(np.array(pcd['points']))
        actions = actions.unsqueeze(0)
        actions = actions.to(dtype=torch.float64)
            
        # remove later
        votes = 10
        actions_tile = actions.repeat(votes, 1)
        pos_noise = torch.empty(votes, 3).uniform_(-0.005, 0.005)
        actions_tile[:, 3:6] = actions_tile[:, 3:6] + pos_noise

        points, actions = points.type(torch.FloatTensor).cuda(), actions.type(torch.FloatTensor).cuda()
        actions_tile = actions_tile.type(torch.FloatTensor).cuda()
        points = points.transpose(2, 1)
        
        # # ---------------------------
        # for _ in range(vote_num):
        #     pred, _ = classifier(points, actions)
        #     # vote_pool += pred
        # # pred = vote_pool / vote_num
        # # pred_choice = pred.data.max(1)[1]

        # probabilities = torch.sigmoid(pred)
        # # Convert probabilities to binary predictions (0 or 1)
        # pred_choice = (probabilities >= 0.5).float()

        # print(f"target: pred_choice, pred_prob: ", target.item(), pred_choice.item(), probabilities.item())
        # # -----------------------------
        
        preds, pred_choices = [], []
        # TODO: Vectorize this
        for i in range(votes):
            pred, _ = self.classifier(points, actions_tile[i:i+1])
            preds.append(pred)
            probabilities = torch.sigmoid(pred)
            # Changed from 0.5
            pred_choice = (probabilities >= 0.5).float()
            pred_choices.append(pred_choice.item())
    

        count_ones = pred_choices.count(1.0)
        count_zeros = pred_choices.count(0.0)
        if count_ones > count_zeros:
            pred_choice = torch.tensor(1.0).cuda()
            confidence = count_ones
        elif count_zeros > count_ones:
            pred_choice = torch.tensor(0.0).cuda()
            confidence = count_zeros
        else:
            pred_choice = torch.tensor(1.0).cuda()
            confidence = count_ones

        print(f"pred_choice, pred_prob: ", pred_choice.item(), confidence)
        # print(f"target: pred_choice, pred_prob: ", target.item(), pred_choice.item(), confidence)
        # collision_count += target.item()
        # if target.item() == 1.0 and pred_choice.item() == 0.0:
        #     false_negative += 1
        # if target.item() == 0.0 and pred_choice.item() == 1.0:
        #     false_positive += 1
        # print("total collisions, false_negative, false_positive: ", collision_count, false_negative, false_positive)

        # pcd = o3d.geometry.PointCloud()
        # # Assign the points to the PointCloud object
        # pcd.points = o3d.utility.Vector3dVector(org_points[0])
        # # Visualize the point cloud
        # o3d.visualization.draw_geometries([pcd])
        
        # for cat in np.unique(target.cpu()):
        #     classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
        #     class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
        #     class_acc[cat, 1] += 1
        # correct = pred_choice.eq(target.long().data).cpu().sum()
        # mean_correct.append(correct.item() / float(points.size()[0]))

        return pred_choice.item()


