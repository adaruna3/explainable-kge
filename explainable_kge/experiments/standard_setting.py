import numpy as np
from copy import copy
import torch
import os
import __main__  # used to get the original execute module

from explainable_kge.models import model_utils
from explainable_kge.logger.terminal_utils import logout, log_train, log_test, load_config
from explainable_kge.logger.viz_utils import ProcessorViz, AbstractProcessorViz

import pdb


def setup_experiment(args):
    # init batch processors for training and validation
    train_args = copy(args)
    train_args["dataset"]["set_name"] = str(args["logging"]["log_num"]) + "_train2id"
    train_args["continual"]["session"] = str(args["logging"]["log_num"])
    tr_bp = model_utils.TrainBatchProcessor(train_args)
    dev_args = copy(args)
    dev_args["dataset"]["set_name"] = str(args["logging"]["log_num"]) + "_valid2id"
    dev_args["continual"]["session"] = str(args["logging"]["log_num"])
    dev_args["dataset"]["neg_ratio"] = 0
    dev_args["dataset"]["dataset_fps"] = None
    de_bp = model_utils.DevBatchProcessor(dev_args)

    # generate training visualization logging
    if args.sess_mode == "TRAIN":
        viz_args = copy(args)
        viz_args["logging"]["tag"] = os.path.basename(__main__.__file__).split(".")[0]
        viz = ProcessorViz(viz_args)
    else:
        viz_args = copy(args)
        viz_args["logging"]["tag"]= os.path.basename(__main__.__file__).split(".")[0]
        viz = AbstractProcessorViz(viz_args)

    # initializes a single model and optimizer used by all batch processors
    model_optim_args = copy(args)
    model_optim_args["model"]["num_ents"] = len(tr_bp.dataset.e2i)
    model_optim_args["model"]["num_rels"] = len(tr_bp.dataset.r2i)
    model = model_utils.init_model(model_optim_args)
    if model_optim_args["cuda"] and torch.cuda.is_available():
        model.to(torch.device("cuda"), non_blocking=True)
    else:
        model.to(torch.device("cpu"), non_blocking=True)
    optimizer = model_utils.init_optimizer(model_optim_args, model)

    tracker_args = copy(args)
    tracker_args["tag"] = os.path.basename(__main__.__file__).split(".")[0]
    tracker_args["sess"] = str(args["logging"]["log_num"])
    tracker = model_utils.EarlyStopTracker(tracker_args)

    return tr_bp, de_bp, viz, model, optimizer, tracker


def setup_test_session(sess, args, model):
    """
    performs pre-testing session operation to load the model
    """
    # loads best model for session
    load_args = copy(args)
    load_args["tag"] = os.path.basename(__main__.__file__).split(".")[0]
    load_args["sess"] = str(sess)
    model = model_utils.load_model(load_args, model)

    return model


if __name__ == "__main__":
    # parse arguments
    exp_config = load_config("Standard setting experiment")

    # select hardware to use
    if exp_config["cuda"] and torch.cuda.is_available():
        logout("Running with CUDA")
    else:
        logout("Running with CPU, experiments will be slow", "w")

    if exp_config["sess_mode"] == "TRAIN":
        exp_tr_bp, exp_de_bp, exp_viz, exp_model, exp_optim, exp_tracker = setup_experiment(exp_config)

        while exp_tracker.continue_training():
            # validate
            if exp_tracker.validate():
                inf_metrics = np.asarray([exp_de_bp.process_epoch(exp_model)])
                # log inference metrics
                exp_viz.add_de_sample(inf_metrics)
                log_label = "i" if exp_tracker.get_epoch() == 0 else "s"
                log_train(inf_metrics, exp_tracker.get_epoch(),
                          0, exp_config["continual"]["num_sess"], log_label,
                          None, None,
                          exp_viz.log_fp, exp_config["logging"]["log_num"])
                # update tracker for early stopping & model saving
                exp_tracker.update_best(0, inf_metrics, exp_model)
            
            # train
            exp_viz.add_tr_sample(0, exp_tr_bp.process_epoch(exp_model, exp_optim))
            exp_tracker.step_epoch()

        # logs the final performance for session (i.e. best)
        best_performance, best_epoch = exp_tracker.get_best()
        log_train(best_performance, best_epoch, 0,
                  exp_config["continual"]["num_sess"], "f", None, None,
                  exp_viz.log_fp, exp_config["logging"]["log_num"])

    elif exp_config["sess_mode"] == "TEST":
        logout("Testing running...", "i")
        exp_tr_bp, exp_de_bp, exp_viz, exp_model, exp_optim, exp_tracker = setup_experiment(exp_config)

        exp_model = setup_test_session(0, exp_config, exp_model)
        inf_metrics = np.asarray([exp_de_bp.process_epoch(exp_model)])
        log_test(inf_metrics, 0, 0,
                 exp_config["continual"]["num_sess"], "f", None, None,
                 exp_viz.log_fp, exp_config["logging"]["log_num"])

    else:
        logout("Mode not recognized for this setting.", "f")
