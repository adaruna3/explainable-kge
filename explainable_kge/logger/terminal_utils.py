from os.path import basename, exists
from argparse import ArgumentParser
from copy import copy
from sys import stdin
from select import select
import torch
import numpy as np
from omegaconf import OmegaConf
import __main__  # used to get the original execute module

import pdb

type2color = {
    's': ' \033[95mSuccess:\033[00m {}',
    'i': ' \033[94mInfo:\033[00m {}',
    'd': ' \033[92mDebug:\033[00m {}',
    'w': ' \033[93mWarning:\033[00m {}',
    'e': ' \033[91mError:\033[00m {}',
    'f': ' \033[4m\033[1m\033[91mFatal Error:\033[00m {}'
}


def logout(msg,p_type=''):
    """ Provides coloring debug printing to terminal """
    if not p_type.lower() in type2color:
        start = type2color['d']
    else:
        start = type2color[p_type.lower()]
    print(start.format(msg))


def log_train(performances, epoch, session, num_sess, stage, sample_size,
              model_size, log_dir, log_num):
    template_performances = np.zeros((num_sess, 2))
    with open(log_dir+"/performances_" + str(log_num) + ".csv", "a") as f:
        f.write(str(session) + ",")
        f.write(str(stage) + ",")
        f.write(str(epoch) + ",")
        f.write(str(sample_size) + ",")
        f.write(str(model_size) + ",")
        for row in range(template_performances.shape[0]):
            if row < performances.shape[0]:
                for value in performances[row, :]:
                    f.write(str(value) + ",")
            else:
                for value in template_performances[row, :]:
                    f.write(str(value) + ",")
        f.write("\n")


def log_test(performances, epoch, session, num_sess, stage, sample_size,
             model_size, log_dir, log_num):
    template_performances = np.zeros((num_sess, 2))
    with open(log_dir+"/test_" + str(log_num) + ".csv", "a") as f:
        f.write(str(session) + ",")
        f.write(str(stage) + ",")
        f.write(str(epoch) + ",")
        f.write(str(sample_size) + ",")
        f.write(str(model_size) + ",")
        for row in range(template_performances.shape[0]):
            if row < performances.shape[0]:
                for value in performances[row, :]:
                    f.write(str(value) + ",")
            else:
                for value in template_performances[row, :]:
                    f.write(str(value) + ",")
        f.write("\n")


def load_config(description):
    # parse for cmd line for config file
    parser = ArgumentParser(description=description)
    parser.add_argument("--config_file", help='config yaml file', type=str, required=True)
    parser.add_argument("--options", help='config overrides', type=str, default="")
    exp_args = parser.parse_args()
    try:
        check = exists(exp_args.config_file)
    except TypeError:
        check = False
    if not check: 
        logout("YAML '{}' does not exist.".format(exp_args.config_file), "f")
        exit()
    # load the config file variables with user confirmation
    exp_config = OmegaConf.load(exp_args.config_file)
    if exp_args.options != "":
        options = eval(exp_args.options)
        exp_config = override_options(exp_config, options)
    if exp_config.sess_mode:
        flag = "non-test"
    else:
        flag = "test"
    logout("The current " + flag + "ing parameters are: \n" + str(exp_config),"i")
    if not confirm_args():
        exit()

    return exp_config

def confirm_args():
    """ captures input from user for 10 s, if no input provided returns """
    logout('Continue training? (Y/n) waiting 10s ...',"i")
    i, o, e = select([stdin], [], [], 1.0)
    if i:  # read input
        cont = stdin.readline().strip()
        if cont == 'Y' or cont == 'y' or cont == '':
            return True
        else:
            return False
    else:  # no input, start training
            return True
        
def override_options(config, options):
    """ recursively parse config options for overrides to defaults """
    for opt_key, opt_value in options.items():
        assert opt_key in config, "Config missing option key " + opt_key
        if type(opt_value) is not dict:
            config[opt_key] = opt_value
        else:
            config[opt_key] = override_options(config[opt_key], opt_value)            
    return config


class InteractiveTerminal:
    """ enables user to interactively query the graph-embedding """
    def __init__(self, args):
        # TODO fix the missing symbols sometime...
        self.args = copy(args)
        # loads a dataset to compare with the model
        self.ground_truth = data_utils.TripleDataset(self.args.dataset, self.args.neg_ratio)
        self.ground_truth.load_counts("gt2id.txt")
        self.train_data = data_utils.TripleDataset(self.args.dataset, self.args.neg_ratio)
        self.train_data.load_counts("train2id.txt")
        # loads a model for querying
        model_args = copy(self.args)
        model_args.num_ents = len(self.ground_truth.e2i)
        model_args.num_rels = len(self.ground_truth.r2i)
        self.model = model_utils.init_model(model_args)
        load_args = copy(self.args)
        load_args.tag = basename(__main__.__file__).split(".")[0]
        load_args.sess = self.args.num_sess - 1
        self.model = model_utils.load_model(load_args, self.model)

    def query(self):
        """ queries the model and ground truth with a triple, displaying the results """
        # gets the query from the user or parameters
        h, r, t = self.capture_query()

        # get ranks for the query
        h_ranked, r_ranked, t_ranked = self.get_ranks(h, r, t)

        # output to terminal if query was made non-programatically
        self.show_title(h, r, t)
        self.show_results(h_ranked, "( _, r, t )")
        self.show_results(r_ranked, "( h, _, t )", 5)
        self.show_results(t_ranked, "( h, r, _ )")

        # retruns ranks for programatically evokeed queries
        return h_ranked, r_ranked, t_ranked

    def continue_interaction(self):
        """ checks whether the wants to continue making queries """
        not_done = True
        response_str = input("\n\nQuery again? (Y/n)")
        if (response_str == "N") or (response_str == "n"):
            not_done = False
        return not_done

    def capture_query(self):
        """ get the query input from the user """
        h = None
        r = None
        t = None
        invalid = True
        while invalid:
            head_str = input("Please enter the head.\n")
            try:
                h = self.ground_truth.e2i[head_str]
                invalid = False
            except KeyError:
                logout(head_str + " is not a valid head.", "e")

        invalid = True
        while invalid:
            relation_str = input("Please enter the relation.\n")
            try:
                r = self.ground_truth.r2i[relation_str]
                invalid = False
            except KeyError:
                logout(relation_str + " is not a valid relation.", "e")

        invalid = True
        while invalid:
            tail_str = input("Please enter the tail.\n")
            try:
                t = self.ground_truth.e2i[tail_str]
                invalid = False
            except KeyError:
                logout(tail_str + " is not a valid tail.", "e")

        return torch.tensor(h, dtype=torch.long).to(self.args.device), \
               torch.tensor(r, dtype=torch.long).to(self.args.device), \
               torch.tensor(t, dtype=torch.long).to(self.args.device)

    def get_ranks(self, h, r, t):
        """ returns rankings for subset of a triple """
        # array for all ents & rels
        np_ents = np.asarray(list(self.ground_truth.e2i.values()))
        np_rels = np.asarray(list(self.ground_truth.r2i.values()))
        torch_ents = torch.tensor(np_ents, dtype=torch.long).to(self.args.device)
        torch_rels = torch.tensor(np_rels, dtype=torch.long).to(self.args.device)

        # makes queries to model & ground truth for ranks
        model_head_scores = self.model.predict(torch_ents, r, t)
        gt_head_scores = -self.ground_truth.predict(torch_ents, r, t)
        model_rel_scores = self.model.predict(h, torch_rels, t)
        gt_rel_scores = -self.ground_truth.predict(h, torch_rels, t)
        model_tail_scores = self.model.predict(h, r, torch_ents)
        gt_tail_scores = -self.ground_truth.predict(h, r, torch_ents)

        # combines outputs of queries
        ranked_heads = np.stack((np.vectorize(self.ground_truth.i2e.get)(np_ents),
                                 model_head_scores,
                                 gt_head_scores),
                                axis=-1)
        ranked_tails = np.stack((np.vectorize(self.ground_truth.i2e.get)(np_ents),
                                 model_tail_scores,
                                 gt_tail_scores),
                                axis=-1)
        ranked_rels = np.stack((np.vectorize(self.ground_truth.i2r.get)(np_rels),
                                model_rel_scores,
                                gt_rel_scores),
                               axis=-1)

        # order the ranking according to the model's scores
        ranked_heads = ranked_heads[ranked_heads[:, 1].astype(np.float32).argsort()]
        ranked_tails = ranked_tails[ranked_tails[:, 1].astype(np.float32).argsort()]
        ranked_rels = ranked_rels[ranked_rels[:, 1].astype(np.float32).argsort()]

        return ranked_heads, ranked_rels, ranked_tails

    def show_title(self, h, r, t):
        """ displays a title for the query """
        h_str = self.ground_truth.i2e[int(h.cpu().data.numpy())]
        r_str = self.ground_truth.i2r[int(r.cpu().data.numpy())]
        t_str = self.ground_truth.i2e[int(t.cpu().data.numpy())]
        print("Query: " + h_str + " " + r_str + " " + t_str)

    def show_results(self, results, heading, num_lines=10):
        """ displays the list of results for a query, with a subheading """
        print("===========" + heading + "===========")
        for row_idx in range(results.shape[0]):
            if row_idx >= num_lines:
                break
            line = results[row_idx]
            print('{0:<20}  {1:<.2f}  {2:<.2f}'.format(line[0], float(line[1]), float(line[2])))