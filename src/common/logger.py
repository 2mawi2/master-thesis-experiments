import datetime
import os
import csv
import glob
from tensorboardX import SummaryWriter
import shutil
from src.common.utils import *
import json


class Logger:
    def __init__(self, args, scalars: [str, str] = None, algorithm="policy"):
        self.args = args
        if args.logging_path is None:
            self.logging_path = get_logs_dir(algorithm=algorithm)
        else:
            self.logging_path = args.logging_path

        if scalars is None:
            scalars = []

        self.scalars = scalars
        self.write_header = True
        self.log_entry = {}
        self.writer = None
        self.writer_rewards = None

        now = (datetime.datetime.utcnow() - datetime.timedelta(hours=4)).strftime("%b-%d_%H:%M:%S")
        self.summary_writer = SummaryWriter(
            logdir=os.path.join(get_tensorboard_dir(), now))  # save every run in seperate dir

        path = os.path.join(self.logging_path, args.env_name, now)
        os.makedirs(path)
        self.f = open(os.path.join(path, 'log.csv'), 'w')

        with open(os.path.join(path, 'params.json'), 'w+') as fp:
            json.dump(args.__dict__, fp)

    def write(self):
        self.print(self.log_entry)

        if self.write_header:
            self.writer = csv.DictWriter(self.f, fieldnames=[x for x in self.log_entry.keys()])
            self.writer.writeheader()

            self.write_header = False

        self.tf_summarize()
        self.writer.writerow(self.log_entry)
        self.log_entry = {}

    def tf_summarize(self):
        for name, key in self.scalars:
            entry = self.log_entry.get(key)
            if entry is not None:
                self.summary_writer.add_scalar(name, self.log_entry[key], self.log_entry["episode"])

    def print(self, log):
        log_keys = [k for k in log.keys()]
        log_keys.sort()

        print(f'EPISODE {log["episode"]}, Steps = {log["steps"]}, MeanReward = {log["MeanReward"]}')

        for key in log_keys:
            if key[0] != '_':
                print(f'{key:s}: {log[key]:.3g}')
        print('\n')

    def log(self, items):
        self.log_entry.update(items)

    def close(self):
        self.f.close()
