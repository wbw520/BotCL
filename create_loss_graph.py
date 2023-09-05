from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--out_file", type=str, default=None)
    args = parser.parse_args()

    assert args.out_file is not None, "Need --out_file argument"

    return args

if __name__ == "__main__":
    opts = get_args()
    stats = {
        "retri" : [],
        "q" : [],
        "acc" : [],
        "cls" : [],
        "dis" : [],
        "consis" : [],
        "att" : []
    }
    data = None
    with open(opts.out_file) as f:
        lines = f.readlines()
        for line in lines:
            if line[:8] == "Epoch: [":
                parts = line.split("(")
                losses = list(map(lambda x: float(x.split(")")[0]), parts[1:]))

                retri_loss, q_loss, acc, cls_loss, dis_loss, consis_loss, att_loss = losses
                stats

                if data is None:
                    data = np.expand_dims(np.array(losses), axis=0)
                else:
                    data = np.concatenate((data, np.expand_dims(np.array(losses), axis=0)), axis=0)

    print(data.shape)

    x = np.arange(data.shape[0])

    fig = plt.figure()
    plt.plot(x, data[:, 0], label="Retri Loss")
    plt.plot(x, data[:, 1], label="Q Loss")
    plt.plot(x, data[:, 2] * np.max(data), label="Accuracy")
    plt.plot(x, data[:, 3], label="Class Loss")
    plt.plot(x, data[:, 4], label="Dis Loss")
    plt.plot(x, data[:, 5], label="Consistency Loss")
    plt.plot(x, data[:, 6], label="Att Loss")
    plt.legend()
    plt.savefig("loss_graph.png")