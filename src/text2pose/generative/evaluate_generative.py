##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

import os, glob, sys
from tqdm import tqdm
import torch
import numpy as np
import roma
import math
from human_body_prior.body_model.body_model import BodyModel

import text2pose.config as config
from text2pose.data import PVCP2
from text2pose.loss import laplacian_nll, gaussian_nll
from text2pose.utils import mean_std_list, save_to_file, load_from_file
from text2pose.generative.model_generative import PPSG
from text2pose.generative.fid import FID


OVERWRITE_RESULT = False


def eval_model_all_runs(model_path, dataset_version, split='val', num_samples=20):

	# get files for all runs
	files = glob.glob(model_path)
	
	# get results for each run
	all_run_results = {}
	for model_path in files:
		r = eval_model(model_path, dataset_version, split=split, num_samples=num_samples)
		all_run_results = r if not all_run_results else {k:all_run_results[k]+v for k,v in r.items()}

	# average & std over runs
	all_run_results = {k:mean_std_list(v) for k,v in all_run_results.items()}

	return all_run_results, len(files)


def eval_model(model_path, dataset_version, split='val', num_samples=20):
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	suffix = get_epoch_from_model_path(model_path)

	# set seed for reproducibility (sampling for pose generation)
	torch.manual_seed(42)
	np.random.seed(42)
		
	if "posescript-A" in dataset_version:
		# average over captions
		results = {}
		nb_caps = len(config.caption_files[dataset_version])
		get_res_file = lambda cap_ind: os.path.join(os.path.dirname(model_path), f"result_{split}_{dataset_version}_{cap_ind}_{suffix}.txt")
		# load model if results for at least one caption is missing
		if OVERWRITE_RESULT or sum([not os.path.isfile(get_res_file(cap_ind)) for cap_ind in range(nb_caps)]):
			model, text_encoder_name = load_model(model_path, device)
		# compute or load results for the given run & caption
		for cap_ind in range(nb_caps):
			filename_res = get_res_file(cap_ind)
			if not os.path.isfile(filename_res) or OVERWRITE_RESULT:
				d = PVCP2(split=split, text_encoder_name=text_encoder_name)
				cap_results = compute_eval_metrics(model, d, device, num_samples)
				best, avg, dis = compute_sampled_metrics(model, d, device, num_samples)
				for k in best:
					cap_results[f'sample_best_{k}'] = best[k]
					cap_results[f'sample_avg_{k}'] = avg[k]
				cap_results['joint_dis'] = dis['joint_dis']
				cap_results['mesh_dis'] = dis['mesh_dis']
				cap_results['joint_dis_pa'] = dis['joint_dis_pa']
				cap_results['mesh_dis_pa'] = dis['mesh_dis_pa']
				cap_results['joint_dis_log'] = dis['joint_dis_log']
				cap_results['mesh_dis_log'] = dis['mesh_dis_log']
				cap_results['joint_dis_log_pa'] = dis['joint_dis_log_pa']
				cap_results['mesh_dis_log_pa'] = dis['mesh_dis_log_pa']
				save_to_file(cap_results, filename_res)
			else:
				cap_results = load_from_file(filename_res)
			# aggregate results
			results = {k:[v] for k, v in cap_results.items()} if not results else {k:results[k]+[v] for k,v in cap_results.items()}
		results = {k:sum(v)/nb_caps for k,v in results.items()}
		# save average result
		filename_res_avg = os.path.join(os.path.dirname(model_path), f"result_{split}_{dataset_version}_avg_{suffix}.txt")
		save_to_file(results, filename_res_avg)
	else:
		filename_res = os.path.join(os.path.dirname(model_path), f"result_{split}_{dataset_version}_{suffix}.txt")
		# compute or load results
		if not os.path.isfile(filename_res) or OVERWRITE_RESULT:
			model, text_encoder_name = load_model(model_path, device)
			d = PVCP2(split=split, text_encoder_name=text_encoder_name)
			results = compute_eval_metrics(model, d, device, num_samples)
			best, avg, dis = compute_sampled_metrics(model, d, device, num_samples)
			for k in best:
				results[f'sample_best_{k}'] = best[k]
				results[f'sample_avg_{k}'] = avg[k]
    
			results['joint_dis'] = dis['joint_dis']
			results['mesh_dis'] = dis['mesh_dis']
			results['joint_dis_pa'] = dis['joint_dis_pa']
			results['mesh_dis_pa'] = dis['mesh_dis_pa']
			results['joint_dis_log'] = dis['joint_dis_log']
			results['mesh_dis_log'] = dis['mesh_dis_log']
			results['joint_dis_log_pa'] = dis['joint_dis_log_pa']
			results['mesh_dis_log_pa'] = dis['mesh_dis_log_pa']
			save_to_file(results, filename_res)
		else:
			results = load_from_file(filename_res)
		
	return {k:[v] for k, v in results.items()}


def load_model(model_path, device):
	
	assert os.path.isfile(model_path), "File {} not found.".format(model_path)
	
	# load checkpoint & model info
	ckpt = torch.load(model_path, device)
	epoch = ckpt['epoch']
	text_encoder_name = ckpt['args'].text_encoder_name
	latentD = ckpt['args'].latentD
	
	# load model
	model = PPSG(text_encoder_name=text_encoder_name, latentD=latentD).to(device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	print("Loaded model from:", model_path)
	print(f"⭐ The currently loaded epoch is [{epoch}]")
	
	return model, text_encoder_name


def compute_eval_metrics(model, dataset, device, num_samples=20):

	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=None, shuffle=False,
		batch_size=1,
		num_workers=8,
		pin_memory=True,
	)

	results = {}
 
	# compute FID
	fid = FID(version='PVCP2', model=model, device=device)
	fid.extract_real_features(data_loader)
	fid.reset_gen_features()
	for batch in tqdm(data_loader, desc='compute FID metrics'):
		if isinstance(batch['caption_tokens'][0], str):  # Qwen
			caption_tokens = batch['caption_tokens']
			caption_lengths = batch['caption_lengths'].to(device)
		else:  # GloVe
			caption_tokens = batch['caption_tokens'].to(device)
			caption_lengths = batch['caption_lengths'].to(device)
			caption_tokens = caption_tokens[:,:caption_lengths.max()]
		with torch.no_grad():
			onepose = model.sample_text_nposes(caption_tokens, caption_lengths, n=num_samples)['pose_body']
		fid.add_gen_features(onepose)
	fid_value = fid.compute()
	results["fid"] = fid_value

	# compute elbos
	body_model = BodyModel(bm_fname = config.SMPL_NEUTRAL_BM, num_betas = config.SMPL_n_betas, model_type='smpl').to(device)
	elbos = {'v2v': 0.0, 'jts': 0.0, 'rot': 0.0}
	for batch in tqdm(data_loader, desc='compute ELBO metrics'):
		poses = batch['pose'].to(device)
		if isinstance(batch['caption_tokens'][0], str):  # Qwen
			caption_tokens = batch['caption_tokens']
			caption_lengths = batch['caption_lengths'].to(device)
		else:  # GloVe
			caption_tokens = batch['caption_tokens'].to(device)
			caption_lengths = batch['caption_lengths'].to(device)
			caption_tokens = caption_tokens[:,:caption_lengths.max()]
		with torch.no_grad():
			output = model.forward(poses, caption_tokens, caption_lengths)
			bm_rec = body_model(pose_body=output['pose_body_pose'][:,1:22].flatten(1,2), # (Jtr[B,52,3], f[13776,3], full_pose[B,156], v[B,6890,3])
								root_orient=output['pose_body_pose'][:,:1].flatten(1,2))
			bm_orig = body_model(pose_body=poses[:,1:22].flatten(1,2),
								root_orient=poses[:,:1].flatten(1,2))
			kld = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(output['q_z'], output['t_z']), dim=[1]))
		elbos['v2v'] += (-laplacian_nll(bm_orig.v, bm_rec.v, model.decsigma_v2v).sum() - kld).detach().item()
		elbos['jts'] += (-laplacian_nll(bm_orig.Jtr, bm_rec.Jtr, model.decsigma_jts).sum() - kld).detach().item()
		elbos['rot'] += (-gaussian_nll(output['pose_body_matrot_pose'].view(-1,3,3), roma.rotvec_to_rotmat(poses.view(-1, 3)), model.decsigma_rot).sum() - kld).detach().item()

	for k in elbos: elbos[k] /= len(dataset)
 
	results.update(elbos)

	# normalize results
	norm_results = {
          			'fid':results['fid'],
					'jts':results['jts']/(len(bm_orig.Jtr[0]) * 3),
					'v2v':results['v2v']/(len(bm_orig.v[0]) * 3),
					'rot':results['rot']/(model.pose_decoder.num_joints * 9),
				}

	return norm_results

def compute_sampled_metrics(model, dataset, device, num_samples=20):
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=None, shuffle=False,
        batch_size=1, 
        num_workers=8,
        pin_memory=True,
    )

    body_model = BodyModel(bm_fname=config.SMPL_NEUTRAL_BM, num_betas=config.SMPL_n_betas, model_type='smpl').to(device)

    total_best = {'mpjpe': 0.0, 'pa_mpjpe': 0.0, 'mpve': 0.0, 'pa_mpve': 0.0}
    total_avg = {'mpjpe': 0.0, 'pa_mpjpe': 0.0, 'mpve': 0.0, 'pa_mpve': 0.0}
    total_dis = {'joint_dis':0.0, 'mesh_dis': 0.0, 'joint_dis_pa':0.0, 'mesh_dis_pa': 0.0, 
                 'joint_dis_log':0.0, 'mesh_dis_log': 0.0, 'joint_dis_log_pa':0.0, 'mesh_dis_log_pa': 0.0}
    
    total_samples = 0  # 样本总数

    for batch in tqdm(data_loader, desc=f'compute pose metrics [K={num_samples}]'):
        poses = batch['pose'].to(device)
        batch_size = poses.shape[0]

        if isinstance(batch['caption_tokens'][0], str):  # Qwen
            caption_tokens = batch['caption_tokens']
            caption_lengths = batch['caption_lengths'].to(device)
            output_samples = model.sample_str_nposes(s=caption_tokens, n=num_samples)

        else:                                           # GloVe
            caption_tokens = batch['caption_tokens'].to(device)
            caption_lengths = batch['caption_lengths'].to(device)
            caption_tokens = caption_tokens[:,:caption_lengths.max()]
            output_samples = model.sample_text_nposes(caption_tokens, caption_lengths, n=num_samples)

        for b in range(batch_size):
            gt_pose = poses[b]
            bm_orig = body_model(  
                pose_body=gt_pose[1:22].unsqueeze(0).reshape(1, -1),
                root_orient=gt_pose[:1].unsqueeze(0).reshape(1, -1)
            ) # gt

            metrics_list = {k: [] for k in total_best}
            
            for i in range(num_samples):
                pred_pose = output_samples['pose_body'][b, i]
                bm_sample = body_model(
                    pose_body=pred_pose[1:22].unsqueeze(0).reshape(1, -1),
                    root_orient=pred_pose[:1].unsqueeze(0).reshape(1, -1)
                )
                sample_metrics = evaluate_mesh(bm_orig, bm_sample)
                for k in metrics_list:
                    metrics_list[k].append(sample_metrics[k])

            for k in total_best:
                total_best[k] += min(metrics_list[k])
                total_avg[k] += sum(metrics_list[k]) / num_samples
            # compute_dispersity
            sample_dis = evaluate_dispersity(bm_orig, bm_sample)   
            total_dis['joint_dis'] += sample_dis['joint_dis']
            total_dis['mesh_dis'] += sample_dis['mesh_dis']
            total_dis['joint_dis_pa'] += sample_dis['joint_dis_pa']
            total_dis['mesh_dis_pa'] += sample_dis['mesh_dis_pa']
            total_dis['joint_dis_log'] += sample_dis['joint_dis_log']
            total_dis['mesh_dis_log'] += sample_dis['mesh_dis_log']
            total_dis['joint_dis_log_pa'] += sample_dis['joint_dis_log_pa']
            total_dis['mesh_dis_log_pa'] += sample_dis['mesh_dis_log_pa']


            total_samples += 1

    total_best = {k: v*1000 / total_samples for k, v in total_best.items()}
    total_avg = {k: v*1000 / total_samples for k, v in total_avg.items()}
    total_dis = {k: v*1000 / total_samples for k, v in total_dis.items()}

    return total_best, total_avg, total_dis

def get_seed_from_model_path(model_path):
	return model_path.split("/")[-2][len("seed"):]


def get_epoch_from_model_path(model_path):
	return model_path.split("_")[-1].split(".")[0]


def rigid_transform_3D(A, B):
    """
    Compute the rigid transformation (scaling, rotation, translation) 
    that aligns point set A to point set B.

    Args:
        A (torch.Tensor): Source point set of shape (N, 3).
        B (torch.Tensor): Target point set of shape (N, 3).

    Returns:
        c (torch.Tensor): Scale factor (scalar).
        R (torch.Tensor): Rotation matrix of shape (3, 3).
        t (torch.Tensor): Translation vector of shape (3,).
    """
    n, dim = A.shape

    # Compute centroids
    centroid_A = torch.mean(A, dim=0)
    centroid_B = torch.mean(B, dim=0)

    # Subtract centroids
    H = torch.matmul((A - centroid_A).T, B - centroid_B) / n

    # SVD decomposition
    U, s, V = torch.linalg.svd(H)
    R = torch.matmul(V.T, U.T)

    # Ensure a right-handed coordinate system
    if torch.det(R) < 0:
        s[-1] = -s[-1]
        V[:, -1] = -V[:, -1]
        R = torch.matmul(V.T, U.T)

    # Compute scale
    varP = torch.var(A, dim=0).sum()
    c = torch.sum(s) / varP

    # Compute translation
    t = -torch.matmul(c * R, centroid_A) + centroid_B

    return c, R, t


def rigid_align(A, B):
    """
    Apply the rigid transformation to align A with B.

    Args:
        A (torch.Tensor): Source point set of shape (N, 3).
        B (torch.Tensor): Target point set of shape (N, 3).

    Returns:
        A2 (torch.Tensor): Transformed source point set aligned with B.
    """
    c, R, t = rigid_transform_3D(A, B)
    A2 = torch.matmul(A, (c * R).T) + t
    return A2



def evaluate_mesh(gt, pred):

    pred_verts = pred.v
    target_verts = gt.v
    
    pred_j3ds = pred.Jtr
    target_j3ds = gt.Jtr
    num_samples = pred_j3ds.shape[0]
    
    if num_samples == 0:
        print("No samples to evaluate.")
        return {}

    # mpve
    pred_verts = pred_verts - pred_j3ds[:, :1, :]
    target_verts = target_verts - target_j3ds[:, :1, :]
    mpve = torch.mean(torch.sqrt(torch.sum(torch.square(pred_verts - target_verts), axis=2)), axis=1)

	# pa-mpve
    pred_verts_pa = []
    for n in range(num_samples):
        pred_verts_pa.append(rigid_align(pred_verts[n], target_verts[n]))
    pred_verts_pa = torch.stack(pred_verts_pa)
    pa_mpve = torch.mean(torch.sqrt(torch.sum(torch.square(pred_verts_pa - target_verts), axis=2)), axis=1)  # (N, )

    # mpjpe
    pred_j3ds = (pred_j3ds - pred_j3ds[:, :1, :])
    target_j3ds = (target_j3ds - target_j3ds[:, :1, :])

    mpjpe = torch.mean(torch.sqrt(torch.sum(torch.square(pred_j3ds - target_j3ds), axis=2)), axis=1)  # (N, )

	# pa-mpjpe
    pred_j3ds_pa = []
    for n in range(num_samples):
        pred_j3ds_pa.append(rigid_align(pred_j3ds[n], target_j3ds[n]))
    pred_j3ds_pa = torch.stack(pred_j3ds_pa)

    pa_mpjpe = torch.mean(torch.sqrt(torch.sum(torch.square(pred_j3ds_pa - target_j3ds), axis=2)), axis=1)  # (N, )

    error_dict = {
        'mpjpe': mpjpe.mean().item(),
        'pa_mpjpe': pa_mpjpe.mean().item(),
        'mpve': mpve.mean().item(),
        'pa_mpve': pa_mpve.mean().item()
    }
    return error_dict

def compute_distances(gt, preds):
    # preds: [K, N, 3], gt: [N, 3]
    diff = preds - gt.unsqueeze(0)  # [K, N, 3]
    dists = torch.norm(diff, dim=2).mean(dim=1)
    return dists

def compute_dispersity(gt, preds):
    dists = compute_distances(gt, preds)
    dispersity = torch.var(dists, unbiased=True).item()
    return dispersity

def compute_log_dispersity(gt, preds, base=10):
    import math
    dists = compute_distances(gt, preds)
    dispersity = torch.var(dists, unbiased=True).item()
    log_disp = math.log(dispersity + 1e-8, base)  
    return log_disp

def evaluate_dispersity(gt, pred):

    pred_verts = pred.v
    target_verts = gt.v
    
    pred_j3ds = pred.Jtr
    target_j3ds = gt.Jtr
    num_samples = pred_j3ds.shape[0]
    
    # mpve
    pred_verts = pred_verts - pred_j3ds[:, :1, :]
    target_verts = target_verts - target_j3ds[:, :1, :]
    mesh_dispersity = compute_dispersity(target_verts, pred_verts)   
    mesh_dispersity_log = math.log(mesh_dispersity + 1, 10)    
    
    # mpjpe
    pred_j3ds = (pred_j3ds - pred_j3ds[:, :1, :])
    target_j3ds = (target_j3ds - target_j3ds[:, :1, :])
    joint_dispersity = compute_dispersity(target_j3ds, pred_j3ds)  
    joint_dispersity_log = math.log(joint_dispersity + 1, 10)  
    
	# pa-mpve
    pred_verts_pa = []
    for n in range(num_samples):
        pred_verts_pa.append(rigid_align(pred_verts[n], target_verts[n]))
    pred_verts_pa = torch.stack(pred_verts_pa)
    mesh_dispersity_pa = compute_dispersity(target_verts, pred_verts_pa)       
    mesh_dispersity_log_pa = math.log(mesh_dispersity_pa + 1, 10)      

    pred_j3ds_pa = []
    for n in range(num_samples):
        pred_j3ds_pa.append(rigid_align(pred_j3ds[n], target_j3ds[n]))
    pred_j3ds_pa = torch.stack(pred_j3ds_pa)
    joint_dispersity_pa = compute_dispersity(target_j3ds, pred_j3ds_pa)
    joint_dispersity_log_pa = math.log(joint_dispersity_pa + 1, 10)
    
    error_dict = {
        'joint_dis': joint_dispersity,
        'mesh_dis': mesh_dispersity,
        'joint_dis_pa': joint_dispersity_pa,
        'mesh_dis_pa': mesh_dispersity_pa,
        'joint_dis_log': joint_dispersity_log,
        'mesh_dis_log': mesh_dispersity_log,
        'joint_dis_log_pa': joint_dispersity_log_pa,
        'mesh_dis_log_pa': mesh_dispersity_log_pa,
    }
    return error_dict
################################################################################

if __name__=="__main__":

	import argparse

	parser = argparse.ArgumentParser(description='Parameters for the demo.')
	parser.add_argument('--model_path', type=str, help='Path to the model (or one of the models, if averaging over several runs; assuming that paths only differ in the seed value).')
	parser.add_argument('--average_over_runs', action="store_true", help="If evaluating different runs of the same model and aggregating the results.")
	parser.add_argument('--dataset', default='PVCP2', type=str,  help="Evaluation dataset.")
	parser.add_argument('--fid', type=str, help='Version of the fid to used for evaluation.')
	parser.add_argument('--split', default="test", type=str, help="Split to evaluate.")
	parser.add_argument('--n_sample', default=20, type=int, help="Number of samples.")
	args = parser.parse_args()

	# compute results
	if args.average_over_runs:
		model_path = config.normalize_model_path(args.model_path, "*")
		ret, nb = eval_model_all_runs(model_path, args.dataset, split=args.split, num_samples=args.n_sample)
	else:
		ret = eval_model(args.model_path, args.dataset, split=args.split, num_samples=args.n_sample)
		nb = 1

	# display results
	if nb == 1:
		fill = lambda key: '%.2f' % ret[key][0]
	else:
		fill = lambda key: "%.2f \\tiny{${\pm}$ %.2f}" % tuple(ret[key])
 
	print("\n<model> | "
		f"fid: {fill('fid')} \n"
		f"avg-mpjpe: {fill('sample_avg_mpjpe')} | joint_dis_log: {fill('joint_dis_log')} | \n"
		f"avg-mpjpe-pa: {fill('sample_avg_pa_mpjpe')} joint_dis_log_pa: {fill('joint_dis_log_pa')} |\n"
		f"avg-mpve:  {fill('sample_avg_mpve')} | mesh_dis_log: {fill('mesh_dis_log')} |  \n"
		f"avg-mpve-pa: {fill('sample_avg_pa_mpve')} | mesh_dis_log_pa: {fill('mesh_dis_log_pa')}"
    )