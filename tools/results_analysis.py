import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def cvt_lists2ndarray(lists):
    '''Convert a list of lists with different lengths to a np.ndarray
    '''
    lens = [len(i) for i in lists]
    print(lens)
    max_len = max(lens)
    out = np.zeros((len(lists), max_len))
    mask = np.arange(max_len) < np.array(lens)[:, None]
    # fill the out array with mask
    out[mask] = np.concatenate(lists)
    return out

def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        if not all_times:
            raise KeyError(
                'Please reduce the log interval in the config so that'
                'interval is less than iterations of one epoch.')
        all_times = np.array(all_times)
        epoch_ave_time = all_times.mean(-1)
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f}')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f}')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print(f'average iter time: {np.mean(all_times):.4f} s/iter')
        print()


def plot_curve(log_dicts, args):
    import pandas as pd
    if args.backend is not None:
        plt.switch_backend(args.backend)
    # sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        if len(args.json_logs) == 1:
            for metric in args.keys:
                legend.append(f'{metric}')
        else:
            for json_log in args.json_logs:
                for metric in args.keys:
                    legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    '''Add a canvas'''
    if 'loss' in metrics[0]:
        fig = plt.figure(figsize=(20,12), dpi=90)
    else:
        fig = plt.figure(figsize=(16,9), dpi=90)
    font_size = args.font_size

    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        values = []
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            if metric not in log_dict[epochs[int(args.start_epoch) - 1]]:
                if 'mAP' in metric:
                    raise KeyError(
                        f'{args.json_logs[i]} does not contain metric '
                        f'{metric}. Please check if "--no-validate" is '
                        'specified when you trained the model.')
                raise KeyError(
                    f'{args.json_logs[i]} does not contain metric {metric}. '
                    'Please reduce the log interval in the config so that '
                    'interval is less than iterations of one epoch.')
            epochs_ = []
            values_ = []
            if 'mAP' in metric:
                if len(epochs) == max(epochs):
                    xs = np.arange(
                        int(args.start_epoch),
                        max(epochs) + 1, int(args.eval_interval))
                else:
                    xs = np.array(epochs)
                ys = []
                for epoch in epochs:
                    ys += log_dict[epoch][metric]
                epochs_.append(xs)
                values_.append(ys)
                ys = np.array(ys)
                ax = plt.gca()
                ax.tick_params(axis = 'both', which='major', labelsize = 21)
                xs_ticks = xs
                if len(xs) > 14:
                    xs_ticks = xs[xs % 4 == 0]
                ax.set_xticks(xs_ticks, fontsize=font_size)
                plt.xlabel('epoch', fontsize=font_size)
                plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
                plt.plot(xs[np.argmax(ys)],ys[np.argmax(ys)], marker='d', markersize=12, label='{0}: {2} (epoch{1})'.format(metric, np.argmax(ys) + 1, np.max(ys)))
            else:
                xs = []
                ys = []
                num_iters_per_epoch = log_dict[epochs[0]]['iter'][-2]
                for epoch in epochs:
                    iters = log_dict[epoch]['iter']
                    if log_dict[epoch]['mode'][-1] == 'val':
                        iters = iters[:-1]
                    xs.append(
                        np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                    ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                plt.tick_params(labelsize=font_size)
                plt.xlabel('iter', fontsize=font_size)
                plt.plot(
                    xs, ys, label=legend[i * num_metrics + j], linewidth=2)
            plt.legend(fontsize=font_size)
            values.append(epochs_)
            values.append(values_)
        plt.grid(True, axis='y')
        if args.title is not None:
            title = '_'.join(args.title.split('/')[:-1])
            plt.title(title, fontsize=font_size)
    flags = [len(values[0]) == len(values[2*x]) for x in range(len(values)//2)]
    assert np.all(np.array(flags))
    valuesf = [values[0]] + [values[2*i+1] for i in range(len(values)//2)]
    if len(values[0]) != 0:
        final_values = np.concatenate(valuesf, axis=0).transpose(1,0)
    if args.out is None:
        plt.show()
    else:
        if args.title is not None:
            out = '/'.join(args.title.split('/')[:-1])
        print(f'save curve to: {os.path.join(out, args.out)}')
        if len(values[0]) != 0:
            pd.DataFrame(final_values, columns=['epochs']+list(metrics)).to_csv(os.path.join(out, 'final_eval_values.txt'), index=False)
        fig.savefig(os.path.join(out, args.out))
        plt.cla()


def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['bbox_mAP'],
        help='the metric that you want to plot')
    parser_plt.add_argument(
        '--start-epoch',
        type=str,
        default='1',
        help='the epoch that you want to start')
    parser_plt.add_argument(
        '--eval-interval',
        type=str,
        default='1',
        help='the eval interval when training')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser_plt.add_argument(
        '--font_size', type=int, default=20, help='font size of legend and axises')
    parser_plt.add_argument('--out', type=str, default=None)


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
        'the average time')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for i,line in enumerate(log_file):
                log = json.loads(line.strip())
                # skip the first training info line
                if i == 0:
                    continue
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts


def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = load_json_logs(json_logs)

    eval(args.task)(log_dicts, args)


if __name__ == '__main__':
    main()
