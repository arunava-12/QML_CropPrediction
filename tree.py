import os

def generate_tree(start_path, exclude_paths=None, prefix=""):
    if exclude_paths is None:
        exclude_paths = set()

    # Normalize exclude paths for comparison
    exclude_paths = {os.path.normpath(p) for p in exclude_paths}

    try:
        items = sorted(os.listdir(start_path))
    except PermissionError:
        return

    for index, item in enumerate(items):
        item_path = os.path.normpath(os.path.join(start_path, item))

        # Skip excluded paths
        if any(item_path.startswith(ex) for ex in exclude_paths):
            continue

        connector = "└── " if index == len(items) - 1 else "├── "
        print(prefix + connector + item)

        if os.path.isdir(item_path):
            extension = "    " if index == len(items) - 1 else "│   "
            generate_tree(item_path, exclude_paths, prefix + extension)


if __name__ == "__main__":
    root_dir = r"E:\vscode\Quantum_CropPredictor"

    exclude = [
        r"E:\vscode\Quantum_CropPredictor\venv"
    ]

    print(root_dir)
    generate_tree(root_dir, exclude)