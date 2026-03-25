import os

def format_size(size_bytes):
    """Convert bytes to human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_bytes / (1024**3):.2f} GB"


def generate_tree(start_path, exclude_paths=None, prefix=""):
    if exclude_paths is None:
        exclude_paths = set()

    exclude_paths = {os.path.normpath(p) for p in exclude_paths}

    try:
        items = sorted(os.listdir(start_path))
    except PermissionError:
        return

    visible_items = []
    for item in items:
        item_path = os.path.normpath(os.path.join(start_path, item))
        if not any(item_path.startswith(ex) for ex in exclude_paths):
            visible_items.append(item)

    for index, item in enumerate(visible_items):
        item_path = os.path.join(start_path, item)

        connector = "└── " if index == len(visible_items) - 1 else "├── "

        # Get size
        try:
            if os.path.isfile(item_path):
                size = format_size(os.path.getsize(item_path))
                print(prefix + connector + f"{item} ({size})")
            else:
                print(prefix + connector + item + "/")
        except Exception:
            print(prefix + connector + item)

        # Recurse into directories
        if os.path.isdir(item_path):
            extension = "    " if index == len(visible_items) - 1 else "│   "
            generate_tree(item_path, exclude_paths, prefix + extension)


if __name__ == "__main__":
    root_dir = r"E:\vscode\Quantum_CropPredictor"

    exclude = [
        r"E:\vscode\Quantum_CropPredictor\venv",
        r"E:\vscode\Quantum_CropPredictor\.git"
    ]

    print(root_dir)
    generate_tree(root_dir, exclude)