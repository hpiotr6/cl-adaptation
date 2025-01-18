import importlib
import os
import re
import time
from functools import partial, reduce

import hydra
import numpy as np
import torch
import torch.multiprocessing
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, OmegaConf

import src.approach
import src.utils
from src.datasets.data_loader import get_loaders
from src.last_layer_analysis import last_layer_analysis
from src.loggers.exp_logger import MultiLogger
from src.metrics import cm
from src.networks import allmodels, set_tvmodel_head_var, tvmodels
from src.regularizers import NullVarCovRegLoss, VarCovRegLoss

torch.multiprocessing.set_sharing_strategy("file_system")
load_dotenv(find_dotenv())


def add_and_assert_cfg(cfg):
    if cfg.training.vcreg:
        assert (
            cfg.training.vcreg.var_weight is not None
            or cfg.training.vcreg.cov_weight is not None
            or cfg.training.vcreg.reg_layers is not None
        ), "Define vcreg params"

    cfg.training.setdefault("select_best_model_by_val_loss", True)


def collect_layers(model: torch.nn.Module, cfg):
    compiled_pattern = re.compile(cfg.training.vcreg.reg_layers)
    matched_layers = [
        (name, module)
        for name, module in model.named_modules()
        if re.match(compiled_pattern, name)
    ]

    if not len(matched_layers):
        raise ValueError(
            f"No layers matching the pattern '{cfg.training.vcreg.reg_layers}' were found."
        )

    return matched_layers


def construct_varcov_loss(cfg):
    if not cfg.training.vcreg:
        return NullVarCovRegLoss(
            # scale=cfg.training.vcreg.scale,
        )
    return VarCovRegLoss(
        cfg.training.vcreg.var_weight,
        cfg.training.vcreg.cov_weight,
        collect_layers=partial(collect_layers, cfg=cfg),
        delta=cfg.training.vcreg.smooth_cov,
        scale=cfg.training.vcreg.scale,
        n_first_task=cfg.training.vcreg.n_first_task,
    )


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    tstart = time.time()
    OmegaConf.set_struct(cfg, False)

    cfg.misc.results_path = os.path.expanduser(cfg.misc.results_path)

    add_and_assert_cfg(cfg)

    varocov_regularizer = construct_varcov_loss(cfg)

    if cfg.misc.no_cudnn_deterministic:
        print("WARNING: CUDNN Deterministic will be disabled.")
        src.utils.cudnn_deterministic = False

    src.utils.seed_everything(seed=cfg.misc.seed)
    # cfg -- CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.misc.gpu)
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        if cfg.misc.gpu != "cpu":
            raise EnvironmentError("No GPU available")
        print("WARNING: [CUDA unavailable] Using CPU instead!")
        device = "cpu"

    # In case the dataset is too large
    torch.multiprocessing.set_sharing_strategy("file_system")
    # Multiple gpus
    # if torch.cuda.device_count() > 1:
    #     self.C = torch.nn.DataParallel(C)
    #     self.C.to(self.device)
    ####################################################################################################################

    # cfg -- Network
    from src.networks.network import LLL_Net

    if cfg.model.network in tvmodels:  # torchvision models
        tvnet = getattr(
            importlib.import_module(name="torchvision.models"), cfg.model.network
        )
        if cfg.model.network == "googlenet":
            init_model = tvnet(pretrained=cfg.model.pretrained, aux_logits=False)
        else:
            init_model = tvnet(pretrained=cfg.model.pretrained)
        set_tvmodel_head_var(init_model)
    else:  # other models declared in networks package's init
        net = getattr(importlib.import_module(name="src.networks"), cfg.model.network)
        # WARNING: fixed to pretrained False for other model (non-torchvision)
        init_model = net(pretrained=False)

    # cfg -- Continual Learning Approach
    from src.approach.incremental_learning import Inc_Learning_Appr

    Appr = getattr(
        importlib.import_module(name="src.approach." + cfg.training.approach.name),
        "Appr",
    )
    assert issubclass(Appr, Inc_Learning_Appr)

    # cfg -- Exemplars Management
    from src.datasets.exemplars_dataset import ExemplarsDataset

    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)

    # cfg -- GridSearch
    if cfg.training.gridsearch_tasks > 0:
        from src.gridsearch import GridSearch

        gs_cfg, extra_cfg = GridSearch.extra_parser(extra_cfg)
        Appr_finetuning = getattr(
            importlib.import_module(name="src.approach.finetuning"), "Appr"
        )
        assert issubclass(Appr_finetuning, Inc_Learning_Appr)
        GridSearch_ExemplarsDataset = Appr.exemplars_dataset_class()
        print("GridSearch arguments =")
        for arg in np.sort(list(vars(gs_cfg).keys())):
            print("\t" + arg + ":", getattr(gs_cfg, arg))
        print("=" * 108)

    # assert len(extra_cfg) == 0, "Unused cfg: {}".format(" ".join(extra_cfg))
    ####################################################################################################################

    # Log all arguments
    full_exp_name = (
        reduce((lambda x, y: x[0] + y[0]), cfg.data.datasets)
        if len(cfg.data.datasets) > 0
        else cfg.data.datasets[0]
    )
    full_exp_name += "_" + cfg.training.approach.name
    if cfg.misc.exp_name is not None:
        full_exp_name += "_" + cfg.misc.exp_name
    logger = MultiLogger(
        cfg.misc.results_path,
        full_exp_name,
        loggers=cfg.misc.log,
        save_models=cfg.misc.save_models,
        tags=cfg.misc.tags,
        **cfg.wandb,
    )

    # Loaders
    src.utils.seed_everything(seed=cfg.misc.seed)
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(
        cfg.data.datasets,
        cfg.data.num_tasks,
        cfg.data.nc_first_task,
        cfg.data.nc_per_task,
        cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        max_classes_per_dataset=cfg.data.max_classes_per_dataset,
        max_examples_per_class_trn=cfg.data.max_examples_per_class_trn,
        max_examples_per_class_val=cfg.data.max_examples_per_class_val,
        max_examples_per_class_tst=cfg.data.max_examples_per_class_tst,
        extra_aug=cfg.data.extra_aug,
        validation=0.0 if cfg.data.use_test_as_val else 0.1,
    )
    # Apply arguments for loaders
    if cfg.data.use_valid_only:
        tst_loader = val_loader
    if cfg.data.use_test_as_val:
        val_loader = tst_loader
        cfg.training.select_best_model_by_val_loss = False
    max_task = len(taskcla) if cfg.data.stop_at_task == 0 else cfg.data.stop_at_task

    # Network and Approach instances
    src.utils.seed_everything(seed=cfg.misc.seed)
    if len(cfg.data.datasets) > 1:
        raise ValueError(f"Networks are modified to cifar by first occurance")

    net = LLL_Net(
        init_model,
        is_cifar="cifar" in cfg.data.datasets[0],
        remove_existing_head=not cfg.model.keep_existing_head,
    )
    src.utils.seed_everything(seed=cfg.misc.seed)
    # taking transformations and class indices from first train dataset
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
    # appr_kwcfg = {**base_kwcfg, **dict(logger=logger, **cfg.training.approach)}
    if Appr_ExemplarsDataset:
        # appr_kwcfg["exemplars_dataset"] =

        exemplars_dataset = Appr_ExemplarsDataset(
            transform, class_indices, **cfg.data.exemplars
        )
    src.utils.seed_everything(seed=cfg.misc.seed)
    appr = Appr(
        net, device, varocov_regularizer, logger, exemplars_dataset, cfg=cfg.training
    )

    logger.log_args(OmegaConf.to_container(cfg))

    if cfg.training.vcreg:
        hooked_layer_names = [n_m[0] for n_m in collect_layers(net.model, cfg)]
    else:
        hooked_layer_names = []

    appr.varcov_regularizer.hooked_layer_names = hooked_layer_names

    ### Add test loader for oracle evaluation during teacher finetuning
    appr.tst_loader = tst_loader

    # GridSearch
    if cfg.training.gridsearch_tasks > 0:
        raise NotImplementedError
        ft_kwcfg = {
            **base_kwcfg,
            **dict(
                logger=logger,
                exemplars_dataset=GridSearch_ExemplarsDataset(transform, class_indices),
            ),
        }
        appr_ft = Appr_finetuning(net, device, **ft_kwcfg)
        gridsearch = GridSearch(
            appr_ft,
            cfg.misc.seed,
            gs_cfg.gridsearch_config,
            gs_cfg.gridsearch_acc_drop_thr,
            gs_cfg.gridsearch_hparam_decay,
            gs_cfg.gridsearch_max_num_searches,
        )

    # Loop tasks
    print(taskcla)
    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))
    test_loss = np.zeros((max_task, max_task))
    test_var_loss = np.zeros((max_task, max_task))
    test_cov_loss = np.zeros((max_task, max_task))
    test_corr_loss = np.zeros((max_task, max_task))
    test_var_layers = np.zeros((max_task, max_task, len(hooked_layer_names)))
    test_cov_layers = np.zeros((max_task, max_task, len(hooked_layer_names)))
    test_corr_layers = np.zeros((max_task, max_task, len(hooked_layer_names)))

    for t, (_, ncla) in enumerate(taskcla):
        # Early stop tasks if flag
        if t >= max_task:
            continue

        print("*" * 108)
        print("Task {:2d}".format(t))
        print("*" * 108)
        print("\n")

        # Add head for current task
        net.add_head(taskcla[t][1])
        net.to(device)

        # GridSearch
        if t < cfg.training.gridsearch_tasks:
            raise NotImplementedError
            # Search for best finetuning learning rate -- Maximal Plasticity Search
            print("LR GridSearch")
            best_ft_acc, best_ft_lr = gridsearch.search_lr(
                appr.model, t, trn_loader[t], val_loader[t]
            )
            # Apply to approach
            appr.lr = best_ft_lr
            gen_params = gridsearch.gs_config.get_params("general")
            for k, v in gen_params.items():
                if not isinstance(v, list):
                    setattr(appr, k, v)

            # Search for best forgetting/intransigence tradeoff -- Stability Decay
            print("Trade-off GridSearch")
            best_tradeoff, tradeoff_name = gridsearch.search_tradeoff(
                cfg.training.approach.name,
                appr,
                t,
                trn_loader[t],
                val_loader[t],
                best_ft_acc,
            )
            # Apply to approach
            if tradeoff_name is not None:
                setattr(appr, tradeoff_name, best_tradeoff)

            print("-" * 108)

        if t == 0 and cfg.data.ne_first_task is not None:
            appr.nepochs = cfg.data.ne_first_task

        # Train
        if t == 0 and cfg.misc.cache_first_task_model:
            exp_tag = (
                "_".join([d for d in cfg.data.datasets])
                + "_t"
                + str(cfg.data.num_tasks)
                + "s"
                + str(cfg.data.nc_first_task)
            )
            if cfg.data.use_test_as_val:
                exp_tag += "_test_as_val"
            model_tag = cfg.model.network
            if cfg.model.pretrained:
                model_tag += "_pretrained"
            model_tag += (
                "_ep"
                + str(cfg.training.nepochs)
                + "_bs"
                + str(cfg.data.batch_size)
                + "_lr"
                + str(cfg.lr)
                + "_wd"
                + str(cfg.weight_decay)
                + "_m"
                + str(cfg.momentum)
                + "_clip"
                + str(cfg.clipping)
                + "_sched"
                + "_".join([str(m) for m in cfg.scheduler_milestones])
            )
            model_ckpt_dir = os.path.join("checkpoints", exp_tag, model_tag)
            model_ckpt_path = os.path.join(
                model_ckpt_dir, "model_seed_" + str(cfg.misc.seed) + ".ckpt"
            )
            if os.path.exists(model_ckpt_path):
                print("Loading model from checkpoint: " + model_ckpt_path)
                net.load_state_dict(torch.load(model_ckpt_path))
                appr.post_train_process(t, trn_loader[t])
                appr.exemplars_dataset.collect_exemplars(
                    appr.model, trn_loader[t], val_loader[t].dataset.transform
                )
            else:
                appr.train(t, trn_loader[t], val_loader[t])
                print("Saving first task checkpoint to: " + model_ckpt_path)
                os.makedirs(model_ckpt_dir, exist_ok=True)
                torch.save(net.state_dict(), model_ckpt_path)
        else:
            appr.train(t, trn_loader[t], val_loader[t])
        print("-" * 108)

        if t == 0 and cfg.data.ne_first_task is not None:
            appr.nepochs = cfg.training.nepochs

        # Test
        for u in range(t + 1):
            (
                test_loss[t, u],
                test_var_loss[t, u],
                test_cov_loss[t, u],
                test_corr_loss[t, u],
                acc_taw[t, u],
                acc_tag[t, u],
                test_var_layers[t, u],
                test_cov_layers[t, u],
                test_corr_layers[t, u],
            ) = appr.eval(u, tst_loader[u])
            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            print(
                ">>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%"
                "| TAg acc={:5.1f}%, forg={:5.1f}% <<<".format(
                    u,
                    test_loss[t, u],
                    100 * acc_taw[t, u],
                    100 * forg_taw[t, u],
                    100 * acc_tag[t, u],
                    100 * forg_tag[t, u],
                )
            )

        for u in range(max_task):
            logger.log_scalar(
                task=u, iter=t, name="loss", group="test", value=test_loss[t, u]
            )
            logger.log_scalar(
                task=u, iter=t, name="var_loss", group="test", value=test_var_loss[t, u]
            )
            logger.log_scalar(
                task=u, iter=t, name="cov_loss", group="test", value=test_cov_loss[t, u]
            )
            logger.log_scalar(
                task=u,
                iter=t,
                name="corr_loss",
                group="test",
                value=test_corr_loss[t, u],
            )
            logger.log_scalar(
                task=u, iter=t, name="acc_taw", group="test", value=100 * acc_taw[t, u]
            )
            logger.log_scalar(
                task=u, iter=t, name="acc_tag", group="test", value=100 * acc_tag[t, u]
            )

            logger.log_scalar(
                task=u,
                iter=t,
                name="forg_taw",
                group="test",
                value=100 * forg_taw[t, u],
            )
            logger.log_scalar(
                task=u,
                iter=t,
                name="forg_tag",
                group="test",
                value=100 * forg_tag[t, u],
            )

            if not cfg.training.vcreg:
                continue

            for var_val, cov_val, corr_val, layer_name in zip(
                test_var_layers[t, u],
                test_cov_layers[t, u],
                test_corr_layers[t, u],
                hooked_layer_names,
            ):
                logger.log_scalar(
                    task=u,
                    iter=t,
                    name=f"layers_var_loss/{layer_name}",
                    value=var_val.item(),
                    group="test",
                )
                logger.log_scalar(
                    task=u,
                    iter=t,
                    name=f"layers_cov_loss/{layer_name}",
                    value=cov_val.item(),
                    group="test",
                )
                logger.log_scalar(
                    task=u,
                    iter=t,
                    name=f"layers_corr_loss/{layer_name}",
                    value=corr_val.item(),
                    group="test",
                )

        # Save
        print("Save at " + os.path.join(cfg.misc.results_path, full_exp_name))
        logger.log_result(acc_taw, name="acc_taw", step=t, skip_wandb=True)
        logger.log_result(acc_tag, name="acc_tag", step=t, skip_wandb=True)
        logger.log_result(forg_taw, name="forg_taw", step=t, skip_wandb=True)
        logger.log_result(forg_tag, name="forg_tag", step=t, skip_wandb=True)
        if cfg.misc.cm:
            logger.log_result(
                cm(appr.model, tst_loader[: t + 1], cfg.data.num_tasks, appr.device),
                name="cm",
                step=t,
                title="Task confusion matrix",
                xlabel="Predicted task",
                ylabel="True task",
                annot=False,
                cmap="Blues",
                cbar=True,
                vmin=0,
                vmax=1,
            )

        logger.save_model(net.state_dict(), task=t)

        avg_accs_taw = acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1)
        logger.log_result(avg_accs_taw, name="avg_accs_taw", step=t, skip_wandb=True)
        logger.log_scalar(
            task=None,
            iter=t,
            name="avg_acc_taw",
            group="test",
            value=100 * avg_accs_taw[t],
        )
        avg_accs_tag = acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1)
        logger.log_result(avg_accs_tag, name="avg_accs_tag", step=t, skip_wandb=True)
        logger.log_scalar(
            task=None,
            iter=t,
            name="avg_acc_tag",
            group="test",
            value=100 * avg_accs_tag[t],
        )
        aux = np.tril(
            np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0)
        )
        wavg_accs_taw = (acc_taw * aux).sum(1) / aux.sum(1)
        logger.log_result(wavg_accs_taw, name="wavg_accs_taw", step=t, skip_wandb=True)
        logger.log_scalar(
            task=None,
            iter=t,
            name="wavg_acc_taw",
            group="test",
            value=100 * wavg_accs_taw[t],
        )
        wavg_accs_tag = (acc_tag * aux).sum(1) / aux.sum(1)
        logger.log_result(wavg_accs_tag, name="wavg_accs_tag", step=t, skip_wandb=True)
        logger.log_scalar(
            task=None,
            iter=t,
            name="wavg_acc_tag",
            group="test",
            value=100 * wavg_accs_tag[t],
        )

        # Last layer analysis
        if cfg.misc.last_layer_analysis:
            weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True)
            logger.log_figure(name="weights", iter=t, figure=weights)
            logger.log_figure(name="bias", iter=t, figure=biases)

            # Output sorted weights and biases
            weights, biases = last_layer_analysis(
                net.heads, t, taskcla, y_lim=True, sort_weights=True
            )
            logger.log_figure(name="weights", iter=t, figure=weights)
            logger.log_figure(name="bias", iter=t, figure=biases)

    avg_accs_taw = acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1)
    logger.log_result(avg_accs_taw, name="avg_accs_taw", step=0, skip_wandb=False)
    avg_accs_tag = acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1)
    logger.log_result(avg_accs_tag, name="avg_accs_tag", step=0, skip_wandb=False)
    aux = np.tril(
        np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0)
    )
    wavg_accs_taw = (acc_taw * aux).sum(1) / aux.sum(1)
    logger.log_result(wavg_accs_taw, name="wavg_accs_taw", step=0, skip_wandb=False)
    wavg_accs_tag = (acc_tag * aux).sum(1) / aux.sum(1)
    logger.log_result(wavg_accs_tag, name="wavg_accs_tag", step=0, skip_wandb=False)

    # Print Summary
    src.utils.print_summary(acc_taw, acc_tag, forg_taw, forg_tag)
    print("[Elapsed time = {:.1f} h]".format((time.time() - tstart) / (60 * 60)))
    print("Done!")


if __name__ == "__main__":
    main()
