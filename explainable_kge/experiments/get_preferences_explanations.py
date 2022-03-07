import os
from copy import copy, deepcopy
import pickle
from posixpath import split
import numpy as np
import pandas as pd
import torch
import tqdm
import json
import random
import shutil

import __main__  # used to get the original execute module

from explainable_kge.models import model_utils
from explainable_kge.logger import viz_utils
from explainable_kge.models import explain_utils as x_utils
from explainable_kge.logger.terminal_utils import logout, load_config

import pdb


"""
load exp json
filter exp json for train facts the were corrupted
select X% fraction of corrupted train facts
make new dataset with X% fraction of corrupted train facts subsituted for train facts
make new dataset with 100% fraction of corrupted train facts subsituted for train facts
"""

def load_dataset(args):
    ds = model_utils.load_dataset(args)
    ds.load_bernouli_sampling_stats()
    if args["model"]["name"] == "tucker":
        ds.reload_er_vocab()
    ds.load_current_ents_rels()
    ds.load_current_ents_rels()
    ds.model_name = None  # makes __getitem__ only retrieve triples instead of triple pairs
    return ds


def format_xkge_ex(ex):
    ex_str = "I know that "
    for i, part in enumerate(ex["parts"]):
        if i+1 < len(ex["parts"]):
            ex_str += part["str"].strip(".") + ", "
        else:
            if part["str"][0].isupper():
                ex_str += "and " + part["str"][0].lower() + part["str"][1:]
            else:
                ex_str += "and " + part["str"]
    ex_str += " Therefore, it is possible that "
    if ex["str"][0].isupper():
        ex_str += ex["str"][0].lower() + ex["str"][1:]
    else:
        ex_str += ex["str"]
    return ex_str


def format_causal_ex(ex_dict, ex_json):
    rel = ex_json["str_list"][1]
    split_fact = ex_json["fact"].split(",")
    if rel in ["ObjInLoc","ObjOnLoc"]:
        tool_loc = split_fact[-1][:-2].replace("_"," ")
        tool = split_fact[0][:-2].replace("_"," ")
        clean_act = None
    elif rel == "ObjUsedTo":
        tool_loc = None
        tool = split_fact[0][:-2].replace("_"," ")
        clean_act = split_fact[-1][:-2].replace("_"," ")
    elif rel == "HasEffect":
        tool_loc = None
        tool = None
        clean_act = split_fact[0][:-2].replace("_"," ")
    ex_str = ex_dict[rel]
    split_ex_str = ex_str.split(",")
    for i, ex_str_part in enumerate(split_ex_str):
        if ex_str_part == "tool location":
            split_ex_str[i] = tool_loc
        elif ex_str_part == "cleaning tool":
            split_ex_str[i] = tool
        elif ex_str_part == "clean act":
            split_ex_str[i] = clean_act
    ex_str = "".join(split_ex_str)
    return ex_str


def format_history_ex(ds, ex_json):
    triple2d = {
        "scrub.a,HasEffect,clean.s": 2,
        "disinfect.a,HasEffect,clean.s": 4,
        "sponge.a,HasEffect,clean.s": 0,
        "wipe.a,HasEffect,clean.s": 2,
        "dust.a,HasEffect,clean.s": 0,
        "scrubber.o,ObjUsedTo,scrub.a": 0,
        "disinfectant_wipe.o,ObjUsedTo,disinfect.a": 2,
        "washing_sponge.o,ObjUsedTo,sponge.a": 3,
        "cleaning_rag.o,ObjUsedTo,wipe.a": 2,
        "feather_duster.o,ObjUsedTo,dust.a": 1,
        "scrubber.o,ObjOnLoc,sink.l": 2,
        "disinfectant_wipe.o,ObjOnLoc,kitchen_table.l": 4,
        "washing_sponge.o,ObjInLoc,sink.l": 2,
        "cleaning_rag.o,ObjOnLoc,kitchen_counter.l": 2,
        "feather_duster.o,ObjInLoc,cabinet.l": 3,
    }
    d_id = triple2d[ex_json["fact"]]
    d = ds[d_id]
    h,r,t = ex_json["fact"].split(",")
    obs_ti = np.nonzero(d.counts[d.r2i[r],d.e2i[h],:])[0]
    if len(obs_ti) > 3:
        obs_ti = np.random.choice(obs_ti, min(3,len(obs_ti)), replace=False)
    obs_hi = np.nonzero(d.counts[d.r2i[r],:,d.e2i[t]])[0]
    if len(obs_hi) > 3:
        obs_hi = np.random.choice(obs_hi, min(3,len(obs_hi)), replace=False)
    if r == "HasEffect":
        ex_str = "I have not observed that " + ex_json["str"].lower()
        ex_str += " I have observed that the act of "
        for i, hi in enumerate(obs_hi):
            act_list = d.i2e[hi][:-2].split("_")
            act_list[0] = add_ing(act_list[0])
            if i+1 < len(obs_hi) and len(obs_hi) > 2:
                ex_str += " ".join(act_list) + ", "
            elif i+1 < len(obs_hi):
                ex_str += " ".join(act_list) + " "
            else:
                ex_str += "and " + " ".join(act_list)
        ex_str += " an object will make it clean."
    elif r in ["ObjInLoc", "ObjOnLoc"]:
        ex_str = "I have not observed that " + ex_json["str"].lower()
        # other locations of object
        if len(obs_ti):
            ex_str += " I have observed that a " + h[:-2].replace("_"," ") + " is often in "
            for i, ti in enumerate(obs_ti):
                if i+1 < len(obs_ti) and len(obs_ti) > 2:
                    ex_str += "a " + d.i2e[ti][:-2].replace("_"," ") + ", "
                elif i+1 < len(obs_ti):
                    ex_str += "a " + d.i2e[ti][:-2].replace("_"," ") + " "
                else:
                    ex_str += "and a " + d.i2e[ti][:-2].replace("_"," ") + "."
        else:
            # if object not in other locations, mention other objects at location
            if len(obs_hi):
                ex_str += " I have observed that "
                for i, hi in enumerate(obs_hi):
                    if i+1 < len(obs_hi) and len(obs_hi) > 2:
                        ex_str += "a " + d.i2e[hi][:-2].replace("_"," ") + ", "
                    elif i+1 < len(obs_hi):
                        ex_str += "a " + d.i2e[hi][:-2].replace("_"," ") + " "
                    else:
                        ex_str += "and a " + d.i2e[hi][:-2].replace("_"," ")
                ex_str += " are often in a " + t[:-2].replace("_"," ") + "."
    elif r == "ObjUsedTo":
        ex_str = "I have not observed that " + ex_json["str"].lower()
        # other used to
        if len(obs_ti):
            ex_str += " I have observed that a " + h[:-2].replace("_"," ") + " is used to "
            for i, ti in enumerate(obs_ti):
                if i+1 < len(obs_ti) and len(obs_ti) > 2:
                    ex_str += d.i2e[ti][:-2].replace("_"," ") + ", "
                elif i+1 < len(obs_ti):
                    ex_str += d.i2e[ti][:-2].replace("_"," ") + " "
                else:
                    ex_str += "and " + d.i2e[ti][:-2].replace("_"," ") + "."
        else:
            if len(obs_hi):
                ex_str += " I have observed that "
                for i, hi in enumerate(obs_hi):
                    if i+1 < len(obs_hi) and len(obs_hi) > 2:
                        ex_str += "a " + d.i2e[hi][:-2].replace("_"," ") + ", "
                    elif i+1 < len(obs_hi):
                        ex_str += "a " + d.i2e[hi][:-2].replace("_"," ") + " "
                    else:
                        ex_str += "and a " + d.i2e[hi][:-2].replace("_"," ")
                ex_str += " are used to " + t[:-2].replace("_"," ") + "."
    return ex_str


def add_ing(word):
    if word[-1] == "e":
        word = word[:-1]
        return word + "ing"
    elif word[-1] in ["g","b"]:
        return word + word[-1] + "ing"
    elif word in ["put"]:
        return word + "ting"
    elif word in ["mop","drop"]:
        return word + "ping"
    else:
        return word + "ing"


def format_plan(plan, ex_json):
    tool2act = {
        "scrubber.o": "scrub.a",
        "disinfectant_wipe.o": "disinfect.a",
        "washing_sponge.o": "sponge.a",
        "cleaning_rag.o": "wipe.a",
        "feather_duster.o": "dust.a"
    }
    act2tool = {act:tool for tool, act in tool2act.items()}
    # get tool, loc, act
    rel = ex_json["str_list"][1]
    split_fact = ex_json["fact"].split(",")
    if rel in ["ObjInLoc","ObjOnLoc"]:
        tool_loc = split_fact[-1][:-2].replace("_"," ")
        tool = split_fact[0][:-2].replace("_"," ")
        clean_act = tool2act[split_fact[0]][:-2].replace("_"," ")
    elif rel == "ObjUsedTo":
        tool_loc = "kitchen counter"
        tool = split_fact[0][:-2].replace("_"," ")
        clean_act = split_fact[-1][:-2].replace("_"," ")
    elif rel == "HasEffect":
        tool_loc = "kitchen counter"
        tool = act2tool[split_fact[0]][:-2].replace("_"," ")
        clean_act = split_fact[0][:-2].replace("_"," ")
    # format plan with current tool, loc, act
    fmt_plan = copy(plan)
    for i, step in enumerate(fmt_plan):
        step_split = step.split(",")
        for j, step_part in enumerate(step_split):
            if step_part == "tool location":
                step_split[j] = tool_loc
            elif step_part == "cleaning tool":
                step_split[j] = tool
            elif step_part == "clean act":
                step_split[j] = clean_act[0].upper() + clean_act[1:]
        fmt_plan[i] = "".join(step_split)
    # focus plan and add action status
    if rel in ["ObjInLoc","ObjOnLoc"]:
        fmt_plan = fmt_plan[0:3]
    elif rel == "ObjUsedTo":
        fmt_plan = fmt_plan[2:5]
    elif rel == "HasEffect":
        fmt_plan = fmt_plan[4:]
    for i, step in enumerate(fmt_plan):
        if i == 0:
            fmt_plan[i] = [step,"Interrupted Action"]
        elif i == 1:
            fmt_plan[i] = [step,"Next Action"]
        else:
            fmt_plan[i] = [step,"Future Action"]
    return fmt_plan


def setup_experiment(args):
    # load the xkge explanations
    locality_str = str(args["explain"]["locality_k"]) if type(args["explain"]["locality_k"]) == int else "best"
    corrupt_str = "corrupted" if args["explain"]["corrupt_json"] else "clean"
    exp_file = "{}.json".format("explanations_" + exp_config["explain"]["xmodel"] + "_" + exp_config["explain"]["locality"] + "_" + locality_str + "_" + corrupt_str + "_preferences")
    xkge_json = []
    for i in range(5):
        log_folder = args["dataset"]["name"] + "_" + args["model"]["name"] + "_" + str(i)
        fp = os.path.join("explainable_kge/logger/logs", log_folder, "results", exp_file)
        with open(fp,"r") as f:
            xkge_json += json.load(f)
    # load the triple dataset for history-based explanations
    tr_datasets = []
    for i in range(5):
        train_args = copy(args)
        train_args["dataset"]["set_name"] = str(i) + "_train2id"
        train_args["continual"]["session"] = i
        tr_datasets.append(load_dataset(train_args))
        tr_datasets[-1].load_counts(str(i) + "_train2id.txt")
    # init the data strcuture for causal-link-based explanations
    causal_ex = {
        "ObjInLoc": "I will move from the table to the ,tool location, to later grab the ,cleaning tool, from the ,tool location,.",
        "ObjOnLoc": "I will move from the table to the ,tool location, to later grab the ,cleaning tool, from the ,tool location,.",
        "ObjUsedTo": "I will take the ,cleaning tool, to later be able to ,clean act, the table with the ,cleaning tool,.",
        "HasEffect": "I will ,clean act, the table to fulfill the goal of the table being clean."
    }
    # init the data structure fo plans
    plans = [
        "Move to the ,tool location,",
        "Observe ,cleaning tool, at ,tool location,",
        "Grab the ,cleaning tool,",
        "Move to the kitchen table",
        ",clean act, the kitchen table with the ,cleaning tool,",
        "Observe kitchen table clean"
    ]
    return xkge_json, tr_datasets, causal_ex, plans, fp


def store_json(main_fp, exp_json):
    json_fp = os.path.join(main_fp, "rq1_amt_data.json")
    with open(json_fp, "w") as f:
        random.shuffle(exp_json)
        json.dump(exp_json, f)


if __name__ == "__main__":
    # parse arguments
    exp_config = load_config("Standard setting experiment")
    # select hardware to use
    if exp_config["cuda"] and torch.cuda.is_available():
        logout("Running with CUDA")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        exp_config["cuda"] = False
        logout("Running with CPU, experiments will be slow", "w")
    # prepare experiment objects
    xkge_exs, hb_exs, clb_exs, ex_plan, exp_fp = setup_experiment(exp_config)
    
    rq1_exs = []
    # go through each xkge explanation
    for i, xkge_ex in enumerate(xkge_exs):
        rq1_ex = {"responses": []}
        # turn explanation to NL
        rq1_ex["responses"].append(["xkge",format_xkge_ex(xkge_ex)])
        rq1_ex["responses"].append(["causal",format_causal_ex(clb_exs, xkge_ex)])
        #rq1_ex["responses"].append(["history",format_history_ex(hb_exs, xkge_ex)])
        # add plans
        rq1_ex["plan"] = format_plan(ex_plan, xkge_ex)
        # add why questions
        rq1_ex["relation"] = xkge_ex["str_list"][1]
        if rq1_ex["relation"] == "ObjOnLoc":
            rq1_ex["why_q1"] = "Why do you think you will find " + xkge_ex["str_list"][0].lower()
            rq1_ex["why_q1"] += " on " + xkge_ex["str_list"][-1].lower() + "?"
            interrupted_step = str(rq1_ex["plan"][0][0])
            rq1_ex["why_q2"] = "Why will you " + interrupted_step.lower() + "?"
            # add video file
            rq1_ex["video"] = "videos/move.mp4"
        elif rq1_ex["relation"] == "ObjInLoc":
            rq1_ex["why_q1"] = "Why do you think you will find " + xkge_ex["str_list"][0].lower()
            rq1_ex["why_q1"] += " in " + xkge_ex["str_list"][-1].lower() + "?"
            interrupted_step = str(rq1_ex["plan"][0][0])
            rq1_ex["why_q2"] = "Why will you " + interrupted_step.lower() + "?"
            rq1_ex["video"] = "videos/move.mp4"
        elif rq1_ex["relation"] == "ObjUsedTo":
            rq1_ex["why_q1"] = "Why do you think " + xkge_ex["str"][0].lower() + xkge_ex["str"][1:].strip(".") + "?"
            interrupted_step = str(rq1_ex["plan"][0][0])
            rq1_ex["why_q2"] = "Why will you " + interrupted_step[0].lower() + interrupted_step[1:] + "?"
            tool = xkge_ex["fact"].split(",")[0].replace("_"," ")[:-2]
            rq1_ex["video"] = "videos/grab_" + tool + ".mp4"
        elif rq1_ex["relation"] == "HasEffect":
            rq1_ex["why_q1"] = "Why do you think " + xkge_ex["str"][0].lower() + xkge_ex["str"][1:].strip(".") + "?"
            interrupted_step = str(rq1_ex["plan"][0][0])
            rq1_ex["why_q2"] = "Why will you " + interrupted_step[0].lower() + interrupted_step[1:] + "?"
            action = xkge_ex["fact"].split(",")[0].replace("_","")[:-2]
            rq1_ex["video"] = "videos/clean_" + action + ".mp4"
        else:
            logout("Unrecognized relation being explained: " + str(rq1_ex["relation"]), "f")
            exit()
        rq1_exs.append(rq1_ex)
    # store the json
    print(len(rq1_exs))
    store_json(exp_fp, rq1_exs)
    
    
    
