import copy
import os
from time import time

import numpy as np
import torch
from torch.utils import data

from Datasets import NumpyImageDataset
from algo.base import device, MAX_NORM
from algo.base import get_model_params, set_client_from_params, get_acc_loss
from .utils import SummaryWriter
from ALA import ALA


def train_model_fedprox(
    model,
    trn_loader,
    lr,
    epochs,
    weight_decay,
    mu,
    global_params,  # numpy array of length n_par
    n_par,
    sch_step=1,
    sch_gamma=1
):
    """
    FedProx local update:
      minimize  f_i(w) + (mu/2) * || w - w_glob ||^2
    `global_params` must be the SAME ordering & length as get_model_params(model).
    """
    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    # Turn the global numpy vector into a torch tensor once:
    g_par = torch.tensor(global_params, device=device)

    for _ in range(epochs):
        for Xb, yb in trn_loader:
            Xb, yb = Xb.to(device), yb.to(device).long()
            optimizer.zero_grad()

            preds = model(Xb)
            loss  = loss_fn(preds, yb) / Xb.size(0)

            # fetch the freshly concatenated local parameters:
            local_np = get_model_params([model], n_par)[0]         # numpy array
            local_par = torch.tensor(local_np, device=device)     # torch

            # proximal term
            prox = (mu / 2.0) * torch.norm(local_par - g_par)**2

            # total objective
            (loss + prox).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()

        scheduler.step()

    # freeze & eval
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model

def train_model_scaffold(
    model,
    trn_loader,
    lr,
    epochs,
    weight_decay,
    c_global,    
    c_local,     
    sch_step=1,
    sch_gamma=1
):
    model = model.to(device).train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    for _ in range(epochs):
        for Xb, yb in trn_loader:
            Xb, yb = Xb.to(device), yb.to(device).long()
            optimizer.zero_grad()

            preds = model(Xb)
            loss  = loss_fn(preds, yb) / Xb.size(0)
            loss.backward()

            # **apply SCAFFOLD correction per‐parameter**
            with torch.no_grad():
                for p, cg, cl in zip(model.parameters(), c_global, c_local):
                    if p.grad is not None:
                        p.grad.add_(cg.to(device) - cl.to(device))

            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()

        scheduler.step()

    # freeze & eval
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model

def train_model_fedavg(model, trn_loader, learning_rate, epoch, weight_decay, sch_step=1, sch_gamma=1):
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    model.train().to(device)
    for _ in range(epoch):
        for Xb, yb in trn_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = loss_fn(preds, yb.long()) / Xb.size(0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()
        scheduler.step()

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def train_FedAvg(args, data_obj, model_func, init_model):
    act_prob = args.act_prob
    lr = args.learning_rate
    batch_size = args.batch_size
    epoch = args.epoch
    com_amount = args.com_amount
    weight_decay= args.weight_decay
    sch_step = args.sch_step
    sch_gamma = args.sch_gamma
    save_period = args.save_period
    if args.mixed:
        suffix = "mixed_" + args.model_name + "per_" + str(args.personalize) + "_warmup_" + str(args.warmup)
    else:
        suffix = args.aggregation + "_" + args.model_name + "per_" + str(args.personalize)
    result_path = args.result_path

    n_clnt = data_obj.n_client
    client_loaders = data_obj.client_loaders
    client_test_loaders = data_obj.client_test_loaders
    n_par = len(get_model_params([model_func()])[0])
    ## Zaland -- Code for other agg functions
    server_m = None
    server_v = None
    c_global = None
    c_local = None
    if args.aggregation in ['fedavgm', 'fedadam']:
        server_m = np.zeros_like(get_model_params([init_model], n_par)[0])
    if args.aggregation == 'fedadam':
        server_v = np.zeros_like(server_m)
    if args.aggregation == 'scaffold' or args.mixed:
        scaffold_params = [p.clone().detach().zero_() for p in init_model.parameters() if p.requires_grad]
        c_global = [p.clone() for p in scaffold_params]
        c_local = [[p.clone() for p in scaffold_params] for _ in range(n_clnt)]

    weight_list = np.array([len(data_obj.clnt_y[i]) for i in range(n_clnt)]).reshape(n_clnt,1)

    os.makedirs(f"{result_path}Model/{data_obj.name}/{suffix}", exist_ok=True)

    # performance logs
    trn_perf_sel = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2))
    

    init_par = get_model_params([init_model], n_par)[0]
    clnt_params_list = np.tile(init_par, (n_clnt, 1)).astype('float32')

    writer = SummaryWriter(f"{result_path}Runs/{data_obj.name}/{suffix}")

    # ── MIXED ── set up “mixed” mode if enabled
    if args.mixed:
        warmup_steps    = args.warmup
        candidate_algos = ['fedavg','fedprox','scaffold']
        warmup_scores   = {algo: [] for algo in candidate_algos}
        final_algo      = None
    # ── end MIXED ──

    # initialize client models
    clnt_models = []
    for _ in range(n_clnt):
        m = model_func().to(device)
        m.load_state_dict(copy.deepcopy(init_model.state_dict()))
        clnt_models.append(m)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    ala_modules = []
    for cid in range(n_clnt):
        ds = list(zip(data_obj.clnt_x[cid], data_obj.clnt_y[cid]))
        ala_modules.append(ALA(
            cid=cid, loss=loss_fn, train_data=ds,
            batch_size=batch_size, rand_percent=args.rand_percent,
            layer_idx=args.layer_idx, eta=args.eta, device=device
        ))

    # global model
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(init_model.state_dict()))

    for round_idx in range(com_amount):
        # --- client selection ---
        np.random.seed(round_idx)
        act = np.random.rand(n_clnt) <= act_prob
        selected_clnts = np.where(act)[0]
        if len(selected_clnts) == 0:
            selected_clnts = [np.random.randint(n_clnt)]

        print(f"Round {round_idx+1}/{com_amount}, Selected Clients: {selected_clnts}")

        # ── MIXED ── decide which aggregator(s) to test this round
        if args.mixed and round_idx < warmup_steps:
            to_test = candidate_algos
        elif args.mixed and round_idx == warmup_steps:
            # pick best overall from warmup
            avg_acc = {a: np.mean(warmup_scores[a]) for a in candidate_algos}
            final_algo = max(avg_acc, key=avg_acc.get)
            print(f"→ Mixed warmup done, selecting final aggregator: {final_algo}")
            to_test = [final_algo]
        else:
            to_test = [ final_algo if args.mixed else args.aggregation ]
        # ── end MIXED ──

        # --- local update & per‐client eval ---
        train_losses = []; train_accs = []
        test_losses  = []; test_accs  = []
        if args.mixed and round_idx < warmup_steps:
            init_flat = get_model_params([init_model], n_par)[0]   # shape: (n_par,)
            temp_params = {
                'fedavg':   np.tile(init_flat, (n_clnt, 1)),   # shape: (19, n_par)
                'fedprox':  np.tile(init_flat, (n_clnt, 1)),
                'scaffold': np.tile(init_flat, (n_clnt, 1)),
            }

        for cid in selected_clnts:
            # reload client model from global
            base_state = copy.deepcopy(avg_model.state_dict())
            clnt_models[cid] = model_func().to(device)
            clnt_models[cid].load_state_dict(copy.deepcopy(avg_model.state_dict()))
            for p in clnt_models[cid].parameters(): p.requires_grad = True

            # personalization hook
            if args.personalize:
                avg_model.to(device)
                clnt_models[cid].to(device)
                ala_modules[cid].adaptive_local_aggregation(global_model=avg_model,
                                                            local_model=clnt_models[cid])

            trn_loader = client_loaders[f'client_{cid}']

            ### MIx training during warmup -- CHange as incorporating FedProx to FedAvg is tricky
            if args.mixed and round_idx < warmup_steps:
                for algo in candidate_algos:
                    m = model_func().to(device)
                    m.load_state_dict(base_state)
                    if algo == 'fedavg':
                        m = train_model_fedavg(
                            m, trn_loader,
                            lr * (args.lr_decay_per_round**round_idx),
                            epoch, weight_decay, sch_step, sch_gamma
                        )
                    elif algo == 'fedprox':
                        global_np = get_model_params([avg_model], n_par)[0]
                        m = train_model_fedprox(
                            m, trn_loader,
                            lr * (args.lr_decay_per_round**round_idx),
                            epoch, weight_decay,
                            mu=args.fedprox_mu, global_params=global_np,
                            n_par=n_par, sch_step=sch_step, sch_gamma=sch_gamma
                        )
                    else:
                        m = train_model_scaffold(
                            m, trn_loader, 
                            lr * (args.lr_decay_per_round**round_idx),
                            epoch, weight_decay,
                            c_global,
                            c_local[cid], sch_step, sch_gamma
                        )
                    temp_params[algo][cid] = get_model_params([clnt_models[cid]], n_par)[0]
                tm = model_func().to(device)
                tm.load_state_dict(base_state)
                updated = train_model_fedavg(
                    tm,
                    trn_loader, lr * (args.lr_decay_per_round**round_idx),
                    epoch, weight_decay, sch_step, sch_gamma
                )

            ## After warmup
            else:
                algo_to_use = to_test[0]
                print(f"Testing only with algo {algo_to_use}")
                if algo_to_use=='fedavg':
                    updated = train_model_fedavg(
                        clnt_models[cid], trn_loader,
                        lr * (args.lr_decay_per_round**round_idx),
                        epoch, weight_decay, sch_step, sch_gamma
                    )
                elif algo_to_use=='fedprox':
                        global_np = get_model_params([avg_model], n_par)[0]
                        updated = train_model_fedprox(
                            clnt_models[cid], trn_loader,
                            lr * (args.lr_decay_per_round**round_idx),
                            epoch, weight_decay,
                            mu=args.fedprox_mu, global_params=global_np,
                            n_par=n_par, sch_step=sch_step, sch_gamma=sch_gamma
                        )
                else:  # scaffold
                    updated = train_model_scaffold(
                        clnt_models[cid], trn_loader,
                        lr * (args.lr_decay_per_round**round_idx),
                        epoch, weight_decay,
                        c_global, c_local[cid],
                        sch_step, sch_gamma
                    )


            # choose local‐training routine based on to_test (during warmup we pick one)
            # algo_to_use = to_test[0] if len(to_test)==1 else None
            # if algo_to_use is None:
            #     # (during warmup, we _train_ once per client under fedavg
            #     #   but will evaluate all three after parameter collection)
            #     updated = train_model_fedavg(
            #         clnt_models[cid], trn_loader,
            #         lr * (args.lr_decay_per_round**round_idx),
            #         epoch, weight_decay, sch_step, sch_gamma
            #     )
            # else:
            #     if algo_to_use=='fedavg':
            #         updated = train_model_fedavg(
            #             clnt_models[cid], trn_loader,
            #             lr * (args.lr_decay_per_round**round_idx),
            #             epoch, weight_decay, sch_step, sch_gamma
            #         )
            #     elif algo_to_use=='fedprox':
            #         global_np = get_model_params([avg_model], n_par)[0]
            #         updated = train_model_fedprox(
            #             clnt_models[cid], trn_loader,
            #             lr * (args.lr_decay_per_round**round_idx),
            #             epoch, weight_decay,
            #             mu=args.fedprox_mu, global_params=global_np,
            #             n_par=n_par, sch_step=sch_step, sch_gamma=sch_gamma
            #         )
            #     else:  # scaffold
            #         updated = train_model_scaffold(
            #             clnt_models[cid], trn_loader,
            #             lr * (args.lr_decay_per_round**round_idx),
            #             epoch, weight_decay,
            #             c_global, c_local[cid],
            #             sch_step, sch_gamma
            #         )
            clnt_models[cid] = updated
            clnt_params_list[cid] = get_model_params([updated], n_par)[0]

            # per‐client train & test eval
            loss_tr, acc_tr = get_acc_loss(trn_loader, updated, w_decay=0.0)
            train_losses.append(loss_tr);  train_accs.append(acc_tr)
            tst_loader = client_test_loaders[f'client_{cid}']
            loss_te, acc_te = get_acc_loss(tst_loader, updated if args.personalize else avg_model,
                                           w_decay=0.0)
            test_losses.append(loss_te);   test_accs.append(acc_te)

            # print(f"      [Client {cid}] Train→ acc={acc_tr:.4f}, loss={loss_tr:.4f} | "
            #       f"Test→ acc={acc_te:.4f}, loss={loss_te:.4f}")

        # --- if in warmup, evaluate all three purely by aggregation+test and record scores ---
        if args.mixed and round_idx < warmup_steps:
            agg_scores = {}
            w_sel = weight_list[selected_clnts]/weight_list[selected_clnts].sum()
            for algo in candidate_algos:
                flat = np.sum(np.stack(temp_params[algo])[selected_clnts] * w_sel, axis= 0)
                cand = set_client_from_params(model_func(), flat)
                accs = []
                for cid in selected_clnts:
                    _, a = get_acc_loss(client_test_loaders[f'client_{cid}'], cand, w_decay=0.0)
                    accs.append(a)
                agg_scores[algo] = np.mean(accs)

            best = max(agg_scores, key=agg_scores.get)
            warmup_scores[best].append(agg_scores[best])
            print(f"  → Warmup round {round_idx}: best agg {best} @ acc={agg_scores[best]:.4f}")

        # --- average over selected clients for this round ---
        avg_tr_loss = np.mean(train_losses)
        avg_tr_acc  = np.mean(train_accs)
        avg_te_loss = np.mean(test_losses)
        avg_te_acc  = np.mean(test_accs)
        trn_perf_sel[round_idx] = [avg_tr_loss, avg_tr_acc]
        tst_perf_sel[round_idx] = [avg_te_loss, avg_te_acc]
        print(f"**** Round {round_idx+1} Avg Train→ acc={avg_tr_acc:.4f}, loss={avg_tr_loss:.4f}")
        print(f"**** Round {round_idx+1} Avg Test → acc={avg_te_acc:.4f}, loss={avg_te_loss:.4f}")

        # log
        writer.add_scalars('Loss/train_sel',    {'Avg': avg_tr_loss}, round_idx)
        writer.add_scalars('Accuracy/train_sel',{'Avg': avg_tr_acc},  round_idx)
        writer.add_scalars('Loss/test_sel',     {'Avg': avg_te_loss}, round_idx)
        writer.add_scalars('Accuracy/test_sel', {'Avg': avg_te_acc},  round_idx)

        # --- final aggregation using chosen algorithm ---
        # decide final algo this round
        if args.mixed:
            if round_idx < warmup_steps:
                algo_to_use = max(agg_scores, key=agg_scores.get)
            else:
                algo_to_use = final_algo
        else:
            algo_to_use = args.aggregation

        w_sel = weight_list[selected_clnts]/weight_list[selected_clnts].sum()
        if algo_to_use in ['fedavg','fedprox']:
            new_par = np.sum(clnt_params_list[selected_clnts]*w_sel, axis=0)
        elif algo_to_use=='fedavgm':
            delta = np.sum((clnt_params_list[selected_clnts]-init_par)*w_sel, axis=0)
            server_m = args.server_momentum*server_m + (1-args.server_momentum)*delta
            new_par  = init_par + server_m
        elif algo_to_use=='fedadam':
            delta = -np.sum((clnt_params_list[selected_clnts]-init_par)*w_sel, axis=0)
            server_m = args.server_beta1*server_m + (1-args.server_beta1)*delta
            server_v = args.server_beta2*server_v + (1-args.server_beta2)*(delta*delta)
            m_hat = server_m/(1-args.server_beta1**(round_idx+1))
            v_hat = server_v/(1-args.server_beta2**(round_idx+1))
            new_par = init_par - args.server_lr * m_hat/(np.sqrt(v_hat)+args.server_eps)
        else:  # scaffold
            print(f"Aggregating with Scaffold")
            new_par = np.sum(clnt_params_list[selected_clnts]*w_sel, axis=0)
            global_params = [p.clone().detach() for p in avg_model.parameters() if p.requires_grad]
            for cid in selected_clnts:
                client_params = [p.clone.detach() for p in clnt_models[cid].parameters() if p.requires_grad]
                for idx, (c_glob, c_loc, g_p, l_p) in enumerate(zip(c_global, c_local[cid], global_params, client_params)):
                    delta = (l_p - g_p) / (lr * epoch)
                    c_local[cid][idx] = c_loc - c_glob + delta
            for idx in range(len(c_global)):
                stacked = torch.stack([c_local[scid][idx] for scid in selected_clnts], dim=0)
                c_global[idx] = stacked.mean(dim=0)
        avg_model = set_client_from_params(model_func(), new_par)

        # checkpoint
        if (round_idx+1) % save_period == 0:
            torch.save(avg_model.state_dict(),
                       f"{result_path}Model/{data_obj.name}/{suffix}/{round_idx+1}com_sel.pt")

    # --- final per‐client test dump, unchanged ---
    log_dir = os.path.join(result_path, 'Model', data_obj.name, suffix)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'final_client_test_results.csv')
    final_test_acc = []
    with open(log_file, 'w') as lf:
        lf.write('client_id,test_loss,test_acc\n')
        for cid in range(n_clnt):
            tl = client_test_loaders[f'client_{cid}']
            if args.personalize:
                l, a = get_acc_loss(tl, clnt_models[cid], w_decay=0.0)
            else:
                l, a = get_acc_loss(tl, avg_model, w_decay=0.0)
            lf.write(f'{cid},{l:.4f},{a:.4f}\n')
            final_test_acc.append(a)
    print(f'Final per-client test accuracy {np.mean(final_test_acc):.4f}')
    return avg_model, clnt_models, trn_perf_sel, tst_perf_sel
