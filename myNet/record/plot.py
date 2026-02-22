import os
import matplotlib.pyplot as plt


def plot_loss_curve(
    loss_history,
    save_dir,
    filename="loss_curve.png",
    title="Training Loss Curve", 
    xl = "Epoch"
):
    """
    loss_history : list[float]
        Lossの履歴（epoch or step）
    save_dir : str
        ベースとなる保存ディレクトリ
    filename : str
        ファイル名（サブディレクトリを含んでも可）
        例: "log_plot/xxx.png"
    """

    # save_path を先に確定
    save_path = os.path.join(save_dir, filename)

    # ★ 重要：filename に含まれるディレクトリも含めて作成
    save_dir_full = os.path.dirname(save_path)
    if save_dir_full != "":
        os.makedirs(save_dir_full, exist_ok=True)

    epochs = list(range(1, len(loss_history) + 1))

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, loss_history, marker="o", linewidth=2)
    plt.xlabel(xl)
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
