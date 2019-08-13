# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
import numpy as np
from model import RippleNet
from train_util import Early_stop_info, ndcg_at_k
from collections import defaultdict
from time import time
from functools import partial
import pickle
import os

def train(args, data_info, logger):
    train_data, eval_data, test_data = data_info[0], data_info[1], data_info[2]
    n_item, n_user = data_info[3], data_info[4]
    n_entity, n_relation = data_info[5], data_info[6]
    ripple_set = data_info[7]
    if args.show_save_dataset_info:
        print(f'train({len(train_data)}), eval({len(eval_data)}), test({len(test_data)})')

    if args.topk_eval:
        _, eval_record, test_record, topk_data = topk_settings(args, train_data, eval_data, test_data, n_item)

    # create dataset
    train_dataset = get_dataset(train_data, ripple_set, n_hop=args.n_hop, batch_size=args.batch_size)
    eval_dataset = get_dataset(eval_data, ripple_set, n_hop=args.n_hop, batch_size=args.batch_size)
    test_dataset = get_dataset(test_data, ripple_set, n_hop=args.n_hop, batch_size=args.batch_size)
    if args.topk_eval:
        topk_dataset = get_dataset(topk_data, ripple_set, n_hop=args.n_hop, batch_size=args.batch_size)

    # init early stop controller
    early_stop = Early_stop_info(args)

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_fract

    with tf.Session(config=config) as sess:
        model = RippleNet(args, n_entity, n_relation, train_dataset)

        init = tf.global_variables_initializer()
        sess.run(init)

        # load emb from previous stage
        if args.load_emb == True:
            print('load pretrained emb ...')
            model.initialize_pretrained_embeddings(sess)

        for epoch in range(args.n_epoch):
            scores = {t: {} for t in ['train', 'eval', 'test']}

            train_dataset.shuffle(buffer_size=1024)
            model.iter_init(sess, train_dataset)

            # start to train
            t_start = time()
            try:
                while True:
                    model.train(sess)
            except tf.errors.OutOfRangeError:
                pass
            t_flag = time()

            # evaluation
            scores['train'] = evaluation(sess, model, train_dataset)
            scores['eval'] = evaluation(sess, model, eval_dataset)
            scores['test'] = evaluation(sess, model, test_dataset)
            
            early_stop_score = 0.
            if args.topk_eval:
                # topk evaluation
                topk_scores = topk_evaluation(sess, model, topk_dataset, eval_record, test_record, args.k_list)
                for t in ['eval', 'test']:
                    for m in ['p', 'r', 'ndcg']:
                        scores[t][m] = topk_scores[t][m]
                early_stop_score = scores['eval']['r'][-1]
            # else:
            early_stop_score = scores['eval']['auc']

            logger.update_score(epoch, scores)
            
            print('training time: %.1fs' % (t_flag - t_start), end='') 
            print(', total: %.1fs.' % (time() - t_start))

            if early_stop_score >= early_stop.best_score:
                print('save embs ...', end='\r')
                model.save_pretrained_emb(sess)

            if early_stop.update_score(epoch, early_stop_score) == True: break
        
    tf.reset_default_graph()

def get_dataset(data, ripple_set, n_hop=2, batch_size=1024):
    memories = {
        'h': np.array([[ripple_set[user][i][0] for user in data[:, 0]] for i in range(n_hop)], dtype=np.int32),
        'r': np.array([[ripple_set[user][i][1] for user in data[:, 0]] for i in range(n_hop)], dtype=np.int32),
        't': np.array([[ripple_set[user][i][2] for user in data[:, 0]] for i in range(n_hop)], dtype=np.int32)
    }
    inputs = {f'memories_{c}_{i}': memories[c][i] for c in ['h', 'r', 't'] for i in range(n_hop)}
    inputs['users'] = data[:, 0].astype(np.int32)
    inputs['items'] = data[:, 1].astype(np.int32)
    inputs['labels'] = data[:, 2].astype(np.float64)
    return tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size).prefetch(batch_size * 2)

def topk_settings(args, train_data, eval_data, test_data, n_item):
    train_record = get_user_record(train_data)
    eval_record = get_user_record(eval_data)
    test_record = get_user_record(test_data)

    user_list_path = f'{args.path.misc}user_list_{args.n_user_eval}.pickle'
    pop_item_path = f'{args.path.misc}pop_item_{args.n_pop_item_eval}.pickle'

    if not os.path.isfile(user_list_path):
        print('save user list ...')
        user_list = list(set(train_record.keys()) & set(eval_record.keys()) & set(test_record.keys()))
        user_counter = { u: len(train_record[u]) for u in user_list }
        user_counter_sorted = sorted(user_counter.items(), key=lambda x: x[1], reverse=True)
        user_list = [u for u, _ in user_counter_sorted[:args.n_user_eval]]

        with open(user_list_path, 'wb') as f:
            pickle.dump(user_list, f)
    else:
        print('load user list ...')
        with open(user_list_path, 'rb') as f:
            user_list = pickle.load(f)

    with open(pop_item_path, 'rb') as f:
        item_set = set(pickle.load(f))

    data = []
    for user in user_list:
        data += [[user, item, 1] for item in (item_set - train_record[user])]
    data = np.array(data)
    
    return train_record, eval_record, test_record, data

def evaluation(sess, model, dataset):
    auc_list, acc_list, f1_list = [], [], []

    model.iter_init(sess, dataset)
    try:
        while True:
            auc, acc, f1 = model.eval(sess)
            auc_list.append(auc)
            acc_list.append(acc)
            f1_list.append(f1)
    except tf.errors.OutOfRangeError:
        pass

    return {
        'auc': float(np.mean(auc_list)),
        'acc': float(np.mean(acc_list)),
        'f1': float(np.mean(f1_list))
    }

def topk_evaluation(sess, model, dataset, eval_user_dict, test_user_dict, k_list):
    topk_scores = {
       t: {
           m: {
               k: [] for k in k_list
           } for m in ['p', 'r', 'ndcg']
       } for t in ['eval', 'test']
    }
    user_item_score_map = defaultdict(list)
    
    model.iter_init(sess, dataset)
    try:
        while True:
            users, items, scores = model.get_scores(sess)
            for u, i, s in zip(users, items, scores):
                user_item_score_map[u].append((i, s))    
    except tf.errors.OutOfRangeError:
        pass

    for u, item_score_pair in user_item_score_map.items():
        item_score_pair_sorted = sorted(item_score_pair, key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        eval_r_hit, test_r_hit = [], []
        for i in item_sorted[:k_list[-1]]:
            eval_r_hit.append(1 if i in eval_user_dict[u] else 0)
            test_r_hit.append(1 if i in test_user_dict[u] else 0)
            
        for k in k_list:
            eval_hit_num = len(set(item_sorted[:k]) & eval_user_dict[u])
            topk_scores['eval']['p'][k].append(eval_hit_num / k)
            topk_scores['eval']['r'][k].append(eval_hit_num / len(eval_user_dict[u]))
            topk_scores['eval']['ndcg'][k].append(ndcg_at_k(eval_r_hit, k))

            test_hit_num = len(set(item_sorted[:k]) & test_user_dict[u])
            topk_scores['test']['p'][k].append(test_hit_num / k)
            topk_scores['test']['r'][k].append(test_hit_num / len(test_user_dict[u]))
            topk_scores['test']['ndcg'][k].append(ndcg_at_k(test_r_hit, k))

    for t in ['eval', 'test']:
        for m in ['p', 'r', 'ndcg']:
            topk_scores[t][m] = [np.around(np.mean(topk_scores[t][m][k]), decimals=4) for k in k_list]

    return topk_scores

def get_user_record(data):
    user_dict = defaultdict(set)
    for user, item, label in data:
        if label == 1:
            user_dict[user].add(item)

    return user_dict
