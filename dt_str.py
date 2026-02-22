import os

def generate_directory_structure_excluding_specific_files(assets_dir="./", output_file_name="directory_structure.txt"):
    current_directory = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    output_file_path = os.path.join(assets_dir, output_file_name)

    # 省略対象から除外したい拡張子
    always_show_exts = {".py", ".cpp", ".c", ".cu", ".h", ".hpp", ".json", ".txt", ".md"}

    # 除外ディレクトリやファイル
    exclude_names = {".vs", "env", "__pycache__", "Dataset", "pcd-dataset", "dt_str.py", "Mine"}

    def write_structure(dir_path, prefix="", depth=0):
        entries = sorted(os.listdir(dir_path))
        entries = [e for e in entries if e not in exclude_names]
        entries_count = len(entries)

        # 最上層は省略しない
        if depth == 0:
            display_entries = entries
            truncated = False
        else:
            # コードファイルは必ず表示
            code_files = [e for e in entries if os.path.splitext(e)[1] in always_show_exts]
            other_entries = [e for e in entries if e not in code_files]

            max_display = 10
            truncated = len(other_entries) > max_display

            # 上から10件＋コードファイルを必ず表示（重複除去）
            display_entries = list(dict.fromkeys(other_entries[:max_display] + code_files))

        for i, entry in enumerate(display_entries):
            entry_path = os.path.join(dir_path, entry)
            is_last = (i == len(display_entries) - 1) and not truncated
            connector = "└── " if is_last else "├── "

            f.write(f"{prefix}{connector}{entry}\n")

            # 再帰的にフォルダを探索
            if os.path.isdir(entry_path):
                new_prefix = prefix + ("    " if is_last else "│   ")
                write_structure(entry_path, new_prefix, depth + 1)

        # 省略表示
        if truncated:
            omitted = len(entries) - len(display_entries)
            if omitted > 0:
                f.write(f"{prefix}└── ...（{omitted}項目省略）\n")

    with open(output_file_path, "w", encoding="utf-8") as f:
        root_name = os.path.basename(current_directory)
        f.write(f"{root_name}/\n")
        write_structure(current_directory, depth=0)

    print(f"Directory structure saved to: {output_file_path}")

# 実行
generate_directory_structure_excluding_specific_files()
