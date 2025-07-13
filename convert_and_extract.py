import os
import pathlib
import shutil
import subprocess

# --- Configuration ---
# Define the root directory of the repository
repo_root = pathlib.Path(__file__).parent.resolve()
# Define the output directory outside the repo
output_dir = repo_root.parent / 'converted_python_code'

# Define the source directories to scan
source_dirs = [
    repo_root / 'algorithmic_trading',
    repo_root / 'Hands-On-Financial-Trading-with-Python-main',
    repo_root / 'Python-for-Algorithmic-Trading-Cookbook-main'
]

def main():
    """Main function to find, convert, and extract all Python code."""
    # Create the output directory if it doesn't exist
    print(f"Creating output directory at: {output_dir}\n")
    output_dir.mkdir(exist_ok=True)

    file_count = 0
    for src_dir in source_dirs:
        if not src_dir.is_dir():
            print(f"Warning: Source directory not found, skipping: {src_dir}")
            continue

        print(f"--- Scanning directory: {src_dir.name} ---")

        # 1. Find and convert all Jupyter notebooks
        for notebook_path in src_dir.rglob('*.ipynb'):
            # Generate a unique output name to avoid collisions
            relative_path = notebook_path.relative_to(src_dir)
            output_stem = str(relative_path.with_suffix('')).replace(os.sep, '_')
            output_py_name = f"{src_dir.name}_{output_stem}.py"
            
            print(f"  Converting: {notebook_path.name} -> {output_py_name}")
            try:
                subprocess.run(
                    [
                        'python',
                        '-m',
                        'jupyter',
                        'nbconvert',
                        '--to',
                        'script',
                        str(notebook_path),
                        '--output',
                        output_py_name,
                        '--output-dir',
                        str(output_dir),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                file_count += 1
            except subprocess.CalledProcessError as e:
                print(f"    ERROR converting {notebook_path.name}: {e.stderr}")

        # 2. Find and copy all existing Python files
        for py_path in src_dir.rglob('*.py'):
            relative_path = py_path.relative_to(src_dir)
            output_stem = str(relative_path.with_suffix('')).replace(os.sep, '_')
            output_py_name = f"{src_dir.name}_{output_stem}.py"

            print(f"  Copying:    {py_path.name} -> {output_py_name}")
            try:
                shutil.copy(py_path, output_dir / output_py_name)
                file_count += 1
            except Exception as e:
                print(f"    ERROR copying {py_path.name}: {e}")
        print("\n")

    print(f"--- Finished ---")
    print(f"Successfully processed {file_count} files.")
    print(f"All Python scripts are located in: {output_dir}")

if __name__ == '__main__':
    main()
