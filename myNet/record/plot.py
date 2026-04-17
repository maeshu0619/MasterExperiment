import os
import matplotlib.pyplot as plt


class PlotMaker():
    def __init__(self, args):
        self.args = args
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../")
        )
        self.save_dir = os.path.join(base_dir, "log", self.args.date, "MyNetwork_train")
        self.num_loss = 14
        self.x_len = 10
        self.y_len = 4
        self.epo_loss_his = [[] for _ in range(self.num_loss)]
        self.epi_loss_his = [[] for _ in range(self.num_loss)]
        self.epo_loss = [0 for _ in range(self.num_loss)]
        self.epi_loss = [0 for _ in range(self.num_loss)]
        self.epo_avg = [0 for _ in range(self.num_loss)]
        self.epi_avg = [0 for _ in range(self.num_loss)]

        self.filename_epo = f"{args.time}_epo"
        self.filename_epi = f"{args.time}_epi"
        self.title_group = [[0, 1, 10], [2, 5, 6], [11, 12, 13], [3, 7], [4, 8, 9]]
        self.group_title = [
            "other", 
            "compression", 
            "surrogate",
            "prun", 
            "add", 
        ]
        self.filename = [
            "", 
            "_geom", 
            "_com", 
            "_prun", 
            "_add", 
            "_single", 
            "_nodes", 
            "_out", 
            "_fit", 
            "_rep", 
            "_disp",
            "_sur_train",
            "_sur_bit_err",
            "_sur_mean_err"
        ]
        self.title = [
            "Loss", 
            "Loss of Geometry", 
            f"Loss of Compression Rate ({getattr(args, 'compression_rate_metric', 'total_bits')})", 
            "Loss of Pruning", 
            "Loss of Adding", 
            "Loss of Single Child Nodes", 
            "Loss of Nodes", 
            "Loss of Outlier", 
            "Loss of Fitting", 
            "Loss of Repulsive", 
            "Loss of Displacing",
            "Surrogate Teacher Fit Loss (SmoothL1)",
            "Surrogate Bit Prediction Error (%)",
            "Surrogate Mean Prediction Error (bit/node/single/bpn, %)"
        ]

    def _plus_loss(self, loss_list, epoORepi):
        if epoORepi == "epo":
            for i in range(len(self.epo_loss_his)):
                self.epo_loss[i] += loss_list[i]
        else:
            for i in range(len(self.epi_loss_his)):
                self.epi_loss[i] += loss_list[i]

    def _reset_loss(self, epoORepi):
        if epoORepi == "epo":
            self.epo_loss = [0 for _ in range(self.num_loss)]
        else:
            self.epi_loss = [0 for _ in range(self.num_loss)]
    
    def _cal_avg(self, length, epoORepi):
        if epoORepi == "epo":
            for i in range(len(self.epo_loss)):
                self.epo_avg[i] = self.epo_loss[i] / length
            self._plus_loss(self.epo_avg, "epi")
            self._append_list(self.epo_loss_his, self.epo_avg)
        else:
            for i in range(len(self.epi_loss)):
                self.epi_avg[i] = self.epi_loss[i] / length
            self._append_list(self.epi_loss_his, self.epi_avg)
    
    def _append_list(self, appended_list, appending_list):
        for i in range(len(appended_list)):
            appended_list[i].append(float(appending_list[i]))

    def plot_loss_curve(self, epoORepi):
        if epoORepi == "epo":
            loss_history = self.epo_loss_his
            filename_front = self.filename_epo
            xl = "Epoch"
        else:
            loss_history = self.epi_loss_his
            filename_front = self.filename_epi
            xl = "Episode"

        os.makedirs(self.save_dir, exist_ok=True)

        # 各損失を個別ファイルでも保存する
        # for loss_idx in range(self.num_loss):
        #     save_path = os.path.join(
        #         self.save_dir,
        #         f"{filename_front}{self.filename[loss_idx]}.png",
        #     )
        #     fig, ax = plt.subplots(1, 1, figsize=(self.x_len, self.y_len))
        #     epochs = list(range(1, len(loss_history[loss_idx]) + 1))
        #     ax.plot(epochs, loss_history[loss_idx], marker="o", linewidth=2)
        #     ax.set_xlabel(xl)
        #     ax.set_ylabel("Loss")
        #     ax.set_title(self.title[loss_idx])
        #     ax.grid(True)
        #     plt.tight_layout()
        #     plt.savefig(save_path)
        #     plt.close(fig)

        for group_idx, group in enumerate(self.title_group):
            save_path = os.path.join(self.save_dir, f"{filename_front}_{self.group_title[group_idx]}.png")
            save_dir_full = os.path.dirname(save_path)
            if save_dir_full != "":
                os.makedirs(save_dir_full, exist_ok=True)

            fig, axes = plt.subplots(len(group), 1, figsize=(self.x_len, self.y_len * len(group)))

            if len(group) == 1:
                axes = [axes]

            for ax, loss_idx in zip(axes, group):
                epochs = list(range(1, len(loss_history[loss_idx]) + 1))
                ax.plot(epochs, loss_history[loss_idx], marker="o", linewidth=2)
                ax.set_xlabel(xl)
                ax.set_ylabel("Loss")
                ax.set_title(self.title[loss_idx])
                ax.grid(True)

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)


    def epi_loss_return(self):
        return self.epi_avg[0]
