import torch

def grad_norm_of(module):
    s = 0.0
    for p in module.parameters():
        if p.grad is not None:
            s += float(p.grad.detach().abs().mean().cpu())
    return s


# def backward_and_measure(tag, loss_term, model, optimizer, writer, args, retain_graph=True):
#     optimizer.zero_grad(set_to_none=True)

#     if torch.is_tensor(loss_term) and loss_term.requires_grad:
#         loss_term.backward(retain_graph=retain_graph)
#     else:
#         writer.write(
#             f"[GradCheck][SKIP] {tag} gradless: "
#             f"type={type(loss_term)}, "
#             f"is_tensor={torch.is_tensor(loss_term)}, "
#             f"requires_grad={getattr(loss_term, 'requires_grad', None)}"
#         )

#     gn_add = grad_norm_of(model.adding_module) if args.add else 0.0
#     gn_pru = grad_norm_of(model.prun_module) if args.prune else 0.0
#     gn_dis = grad_norm_of(model.disp_module) if args.disp else 0.0

#     writer.write(f"[GradCheck] {tag}(add/pru/dis)={gn_add:.3e}/{gn_pru:.3e}/{gn_dis:.3e}")

#     optimizer.zero_grad(set_to_none=True)

def backward_and_measure(tag, loss_term, model, optimizer, writer, args, retain_graph=True):
    # ===== 中間テンソルのgradを消す =====
    def _clear_debug_grads():
        if hasattr(model, "prun_module") and hasattr(model.prun_module, "debug_tensors"):
            for t in model.prun_module.debug_tensors.values():
                if torch.is_tensor(t):
                    t.grad = None

        if hasattr(model, "adding_module") and hasattr(model.adding_module, "debug_tensors"):
            for t in model.adding_module.debug_tensors.values():
                if torch.is_tensor(t):
                    t.grad = None

        if hasattr(model, "debug_tensors"):
            for t in model.debug_tensors.values():
                if torch.is_tensor(t):
                    t.grad = None

    # ===== パラメータ勾配ノルム =====
    def _safe_param_grad_norm(module):
        if module is None:
            return 0.0
        return grad_norm_of(module)

    # ===== 中間テンソル勾配統計 =====
    def _grad_stat(tensor):
        if tensor is None or (not torch.is_tensor(tensor)) or (tensor.grad is None):
            return None, None
        g = tensor.grad.detach()
        mean_abs = g.abs().mean().item()
        nz_ratio = (g.abs() > 1e-12).float().mean().item()
        return mean_abs, nz_ratio

    # ===== 表1行を作る =====
    def _row(name, tensor):
        mean_abs, nz_ratio = _grad_stat(tensor)
        if mean_abs is None:
            return f"| {name:<14} | {'None':>10} | {'None':>10} |"
        return f"| {name:<14} | {mean_abs:>10.3e} | {nz_ratio:>10.3f} |"

    # ===== 表を出力 =====
    def _write_table(title, rows):
        writer.write(f"[GradTable] {title}")
        writer.write("+----------------+------------+------------+")
        writer.write("| target         | mean_abs   | nz_ratio   |")
        writer.write("+----------------+------------+------------+")
        for r in rows:
            writer.write(r)
        writer.write("+----------------+------------+------------+")

    optimizer.zero_grad(set_to_none=True)
    _clear_debug_grads()

    if torch.is_tensor(loss_term) and loss_term.requires_grad:
        loss_term.backward(retain_graph=retain_graph)
    else:
        writer.write(
            f"[GradCheck][SKIP] {tag} gradless: "
            f"type={type(loss_term)}, "
            f"is_tensor={torch.is_tensor(loss_term)}, "
            f"requires_grad={getattr(loss_term, 'requires_grad', None)}"
        )

    # ===== 既存のモジュール単位の勾配 =====
    gn_add = _safe_param_grad_norm(model.adding_module) if args.add else 0.0
    gn_pru = _safe_param_grad_norm(model.prun_module) if args.prune else 0.0
    gn_dis = _safe_param_grad_norm(model.disp_module) if args.disp else 0.0

    writer.write(f"[GradCheck] {tag}(add/pru/dis)={gn_add:.3e}/{gn_pru:.3e}/{gn_dis:.3e}")

    # ===== 削除・追加決定に関する表 =====
    rows = []

    if args.prune and hasattr(model.prun_module, "debug_tensors"):
        dbg = model.prun_module.debug_tensors
        rows.extend([
            _row("prun where", dbg.get("prun_logit")),      # どこを削除するか
            _row("prun how",   dbg.get("keep_ratio_pred")), # どれくらい削除するか
            _row("prun mask",  dbg.get("soft_mask")),       # どこを削除するか
        ])

    if args.add and hasattr(model.adding_module, "debug_tensors"):
        dbg_add = model.adding_module.debug_tensors
        rows.extend([
            _row("add where", dbg_add.get("add_logit")), # どこに追加するか
            _row("add how",   dbg_add.get("mag")),       # どのくらい追加するか
            _row("add dir",   dbg_add.get("dir_vec")),   # どの方向に追加するか
            _row("add mask",   dbg_add.get("add_mask")),   
        ])

    if hasattr(model, "debug_tensors"):
        dbg_net = model.debug_tensors
        rows.extend([
            _row("w prun",  dbg_net.get("keep_w_full")),
            _row("w add",  dbg_net.get("keep_w_full")),
            _row("w final", dbg_net.get("final_w")),      # 下流損失が最終的に見ている重み
        ])

    if len(rows) > 0:
        _write_table(tag, rows)

    optimizer.zero_grad(set_to_none=True)
    _clear_debug_grads()