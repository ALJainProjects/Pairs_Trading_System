import os


def print_directory_structure(startpath):
    """
    Prints the directory structure starting from the given path,
    skipping specified directories like '.venv' and '.git'.
    """
    # Use a set for efficient checking
    excluded_dirs = {'.venv', 'hooks', '.git', '__pycache__'}

    for root, dirs, files in os.walk(startpath, topdown=True):
        # Exclude specified directories by modifying dirs in-place
        # The slice assignment [:] is crucial for os.walk
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level

        # Use os.path.basename to get the folder name
        print(f'{indent}{os.path.basename(root)}/')

        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{sub_indent}{f}')


# Example usage:
# To get the structure of the current directory:
print_directory_structure('.')