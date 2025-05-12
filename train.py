import pickle
import random
import time

from tqdm import tqdm
import numpy as np
import wandb

import torch
import torch.cuda
import torch.nn.functional as F

import config
from models import BigGNN
from utils import cross_entropy, k_fold_by_scene

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)

def train(optimizer, database_3dssg, dataset, batch_size):
    assert(type(dataset) == list)
    indices = [i for i in range(len(dataset))]
    random.shuffle(indices)
    train_loss = 0
    batched_indices = [indices[i:i+batch_size] for i in range(0, len(indices) - batch_size, batch_size)]
    assert(len(batched_indices[0]) == batch_size)
    skipped = 0
    total = 0
    loss_cnt = 0
    for batch in tqdm(batched_indices):
        loss1 = torch.zeros((len(batch), len(batch))).to('cuda')
        loss3 = torch.zeros((len(batch), len(batch))).to('cuda')
        for i in range(len(batch)):
            for j in range(i, len(batch)):
                total += 1
                query = dataset[batch[i]]
                db = database_3dssg[dataset[batch[j]].scene_id]
                query_subgraph, db_subgraph = query, db

                x_node_ft, x_edge_idx, x_edge_ft = query_subgraph.to_pyg()
                p_node_ft, p_edge_idx, p_edge_ft = db_subgraph.to_pyg()
                if len(x_edge_idx[0]) < 1 or len(p_edge_idx[0]) < 1:
                    skipped += 1
                    loss1[i][j] = 1
                    loss1[j][i] = loss1[i][j]
                    loss3[i][j] = 0.5
                    loss3[j][i] = loss3[i][j]
                    continue
                x_p, p_p, m_p = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                                        torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                                        torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))
                x_node_ft, x_edge_idx, x_edge_ft = None, None, None

                loss1[i][j] = 1 - F.cosine_similarity(x_p, p_p, dim=0) # [0, 2] 0 is good
                loss1[j][i] = loss1[i][j]
                loss3[i][j] = m_p
                loss3[j][i] = loss3[i][j]
        loss1_t = (torch.ones((len(batch), len(batch))).to('cuda') - torch.eye(len(batch)).to('cuda')) * 2
        loss3_t = torch.eye(len(batch)).to('cuda')

        # Cross entropy
        loss1 = cross_entropy(loss1, loss1_t, reduction='mean', dim=1)
        loss3 = cross_entropy(loss3, loss3_t, reduction='mean', dim=1)
        if (config.loss_ablation_m): loss = loss1     # Cosine similarity only
        elif (config.loss_ablation_c): loss = loss3   # Matching probability only
        else: loss = (loss1 + loss3) / 2.0          # Average of both

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loss_cnt += 1
    train_loss /= loss_cnt
    return train_loss, loss1.item(), loss3.item()

def eval_loss(database_3dssg, dataset, fold):
    model.eval()
    loss1_across_batches = []
    loss3_across_batches = []
    loss_across_batches = []
    avg_mp_across_batches = []
    avg_mn_across_batches = []
    avg_cos_sim_p_across_batches = []
    avg_cos_sim_n_across_batches = []
    with torch.no_grad():
        assert(type(dataset) == list)
        indices = [i for i in range(len(dataset))]
        random.shuffle(indices)
        batched_indices = [indices[i:i+config.batch_size] for i in range(0, len(indices) - config.batch_size, config.batch_size)]
        assert(len(batched_indices[0]) == config.batch_size)
        skipped = 0
        total = 0
        for batch in batched_indices:
            loss1 = torch.zeros((len(batch), len(batch))).to('cuda')
            loss3 = torch.zeros((len(batch), len(batch))).to('cuda')
            for i in range(len(batch)):
                for j in range(i, len(batch)):
                    total += 1
                    query = dataset[batch[i]]
                    db = database_3dssg[dataset[batch[j]].scene_id]
                    query_subgraph, db_subgraph = query, db
                    
                    x_node_ft, x_edge_idx, x_edge_ft = query_subgraph.to_pyg()
                    p_node_ft, p_edge_idx, p_edge_ft = db_subgraph.to_pyg()
                    if len(x_edge_idx[0]) < 1 or len(p_edge_idx[0]) < 1:
                        skipped += 1
                        loss1[i][j] = 1
                        loss1[j][i] = loss1[i][j]
                        loss3[i][j] = 0.5
                        loss3[j][i] = loss3[i][j]
                        continue
                    x_p, p_p, m_p = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                                            torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                                            torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))
                    x_node_ft, x_edge_idx, x_edge_ft = None, None, None
                    loss1[i][j] = 1 - F.cosine_similarity(x_p, p_p, dim=0) # [0, 2] 0 is good
                    loss1[j][i] = loss1[i][j]
                    loss3[i][j] = m_p
                    loss3[j][i] = loss3[i][j]
            loss1_t = (torch.ones((len(batch), len(batch))).to('cuda') - torch.eye(len(batch)).to('cuda')) * 2
            loss3_t = torch.eye(len(batch)).to('cuda')

            # Average m_p across diagonal
            avg_mp = torch.diag(loss3).mean()
            avg_mn = (torch.sum(loss3) - torch.diag(loss3).sum()) / (len(batch) * (len(batch) - 1))
            avg_cos_sim_p = torch.diag(loss1).mean()
            avg_cos_sim_n = (torch.sum(loss1) - torch.diag(loss1).sum()) / (len(batch) * (len(batch) - 1))
            # Cross entropy
            loss1 = cross_entropy(loss1, loss1_t, reduction='mean', dim=1)
            loss3 = cross_entropy(loss3, loss3_t, reduction='mean', dim=1)
            if (config.loss_ablation_m or config.eval_only_c): loss = loss1     # Use the cosine similarity
            elif (config.loss_ablation_c): loss = loss3   # Use the matching probability only
            else: loss = (loss1 + loss3) / 2.0          # Use the average of both

            loss1_across_batches.append(loss1.item())
            loss3_across_batches.append(loss3.item())
            loss_across_batches.append(loss.item())
            avg_mp_across_batches.append(avg_mp.item())
            avg_mn_across_batches.append(avg_mn.item())
            avg_cos_sim_p_across_batches.append(avg_cos_sim_p.item())
            avg_cos_sim_n_across_batches.append(avg_cos_sim_n.item())

    model.train()
    return torch.tensor(loss_across_batches).mean().item()

def eval_acc(database_3dssg, dataset, eval_iter_count=config.eval_iter_count, out_of=config.out_of, valid_top_k=[1, 3, 5], timer=None):
    model.eval()

    # Make sure the dataset is properly sampled
    buckets = {}
    for idx, g in enumerate(dataset):
        if g.scene_id not in buckets: buckets[g.scene_id] = []
        buckets[g.scene_id].append(idx)

    # Out_of is basically 10
    all_valid = {}
    for _ in range(config.eval_iters):
        valid = {k: [] for k in valid_top_k}

        sampled_test_indices = [[random.sample(buckets[g], 1)[0] for g in random.sample(list(buckets.keys()), out_of)] for _ in range(eval_iter_count)]
        assert(len(sampled_test_indices[0]) == out_of)
        assert(len(sampled_test_indices) == eval_iter_count)
        assert(len(dataset) > 10)

        scene_ids_tset = []
        for t_set in sampled_test_indices:
            true_match = []
            match_prob = []
            cos_sims = []
            scene_ids_tset = []
            for i in t_set:
                query = dataset[t_set[0]]
                db = database_3dssg[dataset[i].scene_id]
                scene_ids_tset.append(db.scene_id)
                assert(query.scene_id == db.scene_id if i == t_set[0] else query.scene_id != db.scene_id)
                query_subgraph, db_subgraph = query, db
                x_node_ft, x_edge_idx, x_edge_ft = query_subgraph.to_pyg()
                p_node_ft, p_edge_idx, p_edge_ft = db_subgraph.to_pyg()

                t1 = time.time()
                x_p, p_p, m_p = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                                        torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                                        torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))
                if timer is not None:
                    timer.text2graph_text_embedding_matching_score_time.append(time.time() - t1)
                    timer.text2graph_text_embedding_matching_score_iter.append(1)

                cos_sims.append((1 - F.cosine_similarity(x_p, p_p, dim=0)).item())
                match_prob.append(m_p.item())
                if (query.scene_id == db.scene_id): true_match.append(1)
                else: true_match.append(0)
            

            if (config.loss_ablation_m or config.eval_only_c):     # Use the cosine similarity only
                # Sort w indices
                cos_sims = np.array(cos_sims) # [0, 2] 0 is good
                true_match = np.array(true_match)
                t1 = time.time()
                sorted_indices = np.argsort(cos_sims)
                sorted_indices = sorted_indices[::-1]
                if timer is not None:
                    timer.text2graph_matching_time.append(time.time() - t1)
                    timer.text2graph_matching_iter.append(1)
                cos_sims = cos_sims[sorted_indices]
                true_match = true_match[sorted_indices]
            elif (config.loss_ablation_c): # Use the matching probability only
                # Sort w indices
                match_prob = np.array(match_prob)
                true_match = np.array(true_match)
                t1 = time.time()
                sorted_indices = np.argsort(match_prob)
                if timer is not None:
                    timer.text2graph_matching_time.append(time.time() - t1)
                    timer.text2graph_matching_iter.append(1)
                match_prob = match_prob[sorted_indices]
                true_match = true_match[sorted_indices]
            else: # Use matching probability only
                # Sort w indices
                match_prob = np.array(match_prob)
                true_match = np.array(true_match)
                t1 = time.time()
                sorted_indices = np.argsort(match_prob)
                if timer is not None:
                    timer.text2graph_matching_time.append(time.time() - t1)
                    timer.text2graph_matching_iter.append(1)
                match_prob = match_prob[sorted_indices]
                true_match = true_match[sorted_indices]

            scene_ids_tset = [scene_ids_tset[i] for i in sorted_indices]

            for k in valid_top_k:
                if (1 in true_match[-k:]): valid[k].append(1)
                else: valid[k].append(0)
        
        for k in valid_top_k:
            if k not in all_valid: all_valid[k] = []
            all_valid[k].append(np.mean(valid[k]))

    accuracy = {k: np.mean(all_valid[k]) for k in valid_top_k}
    print(f'Accuracies: {accuracy}')
    model.train()
    
    return accuracy

def train_with_cross_val(dataset, database_3dssg, folds, epochs, batch_size, entire_training_set):
    if entire_training_set:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        
        starting_epoch = 1
        if (config.continue_training): 
            starting_epoch = config.continue_training
        epochs = epochs + starting_epoch
        for epoch in tqdm(range(starting_epoch, epochs)):
            _, _, _ = train(
                               optimizer=optimizer, 
                               database_3dssg=database_3dssg, 
                               dataset=dataset, 
                               batch_size=batch_size)
            if epoch % config.model_save_epoch == 0:
                torch.save(model.state_dict(), f'{config.model_checkpoints_path}/{config.model_name}_epoch_{epoch}_checkpoint.pt')
        return model
    
    val_losses, accs, durations = [], [], []
    for fold, (train_idx, val_idx) in enumerate(k_fold_by_scene(dataset, folds)):
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        starting_epoch = 1
        if (config.continue_training): 
            starting_epoch = config.continue_training
        epochs = epochs + starting_epoch
        for epoch in tqdm(range(starting_epoch, epochs)):
            train_loss, _, _ = train(
                               optimizer=optimizer, 
                               database_3dssg=database_3dssg, 
                               dataset=train_dataset, 
                               batch_size=batch_size)
            if epoch % config.model_save_epoch == 0:
                torch.save(model.state_dict(), f'{config.model_checkpoints_path}/{config.model_name}_epoch_{epoch}_checkpoint.pt')
            val_losses.append(eval_loss(
                                        database_3dssg=database_3dssg, 
                                        dataset=val_dataset,
                                        fold=fold))
            accs.append(eval_acc(
                                 database_3dssg=database_3dssg, 
                                 dataset=val_dataset,
                                 eval_iter_count=30,
                                 valid_top_k=config.valid_top_k))
            eval_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_losses[-1],
                'val_acc_from_train': accs[-1],
            }
            print(f'Evaluation information: {eval_info}')

            if config.use_wandb:
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_losses[-1],
                })
        if (config.skip_k_fold): break # only use the first fold to speed up training, but we still see a validation

if __name__ == '__main__':
    if (config.model_name is None):
        print("Must define a model name")
        print("Exiting...")
        exit()
    # Make sure only 1 out of 2 loss ablations is true
    if (config.loss_ablation_m and config.loss_ablation_c):
        print("Can only have one loss ablation true at a time")
        print("Exiting...")
        exit()

    if config.use_wandb:
        wandb.config = {"architecture": "Graph Transformer",
                        "dataset": "OSM"}
        wandb_proj_name = f"GOTLoc"
        wandb.init(project=wandb_proj_name,
                    name=config.model_name,
                    mode="online",
                    config=wandb.config)
    
    with open(f"{config.scene_graphs_path}/{config.cell_graphs_file_name}", "rb") as f:
        cell_graphs = pickle.load(f)
    with open(f"{config.scene_graphs_path}/{config.train_text_graphs_file_name}", "rb") as f:
        train_text_graphs = pickle.load(f)
    with open(f"{config.scene_graphs_path}/{config.val_text_graphs_file_name}", "rb") as f:
        val_text_graphs = pickle.load(f)

    train_text_graphs = list(train_text_graphs.values()) # NOTE
    training_set_size = len(train_text_graphs)

    if config.continue_training: 
        model = BigGNN(config.N, config.heads).to('cuda')
        model_dict = torch.load(f'{config.model_checkpoints_path}/{config.continue_training_model}.pt', weights_only=True)
        model.load_state_dict(model_dict)
    else: model = BigGNN(config.N, config.heads).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_with_cross_val(
                            database_3dssg=cell_graphs, 
                            dataset=train_text_graphs,
                            folds=config.folds,
                            epochs=config.epoch,
                            batch_size=config.batch_size,
                            entire_training_set=config.entire_training_set)

    ######### SAVE SOME THINGS #########
    model_name = config.model_name
    args_str = ''
    torch.save(model.state_dict(), f'{config.model_checkpoints_path}/{model_name}.pt')
    # ####################################
    t_start = time.perf_counter()
    # Final test sets evaluation
    test_accuracy = eval_acc(
                                database_3dssg=cell_graphs,
                                dataset=list(val_text_graphs.values()))
    t_end = time.perf_counter()
    print(f'Time elapsed in minutes: {(t_end - t_start) / 60}')
    print(f'Final test set accuracies: {test_accuracy}')