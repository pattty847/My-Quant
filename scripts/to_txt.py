import os

# Define the files and directories to be ignored
ignore_list = [
    # Virtual environments
    ".venv",
    "env",
    "venv",
    "*/env",
    "*/venv",
    "logs",
    "cache",
    "examples",
    
    # Compiled Python files
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "__pycache__",
    "*/__pycache__",
    
    # Version control
    ".git",
    ".svn",
    ".hg",
    
    # IDE and editor settings
    ".cursor",
    ".idea",
    ".vscode",
    "*.sublime-project",
    "*.sublime-workspace",
    
    # OS generated files
    ".DS_Store",
    "Thumbs.db",
    
    # Logs and temporary files
    "*.log",
    "*.tmp",
    "*.bak",
    "*.swp",
    
    # Build directories
    "build",
    "dist",
    "*.egg-info",
    "*.egg",
    
    # Node.js specific
    "node_modules",
    "npm-debug.log",
    
    # Python packaging
    "*.egg-info",
    "pip-wheel-metadata",
    
    # Documentation and metadata
    "README.md",
    "requirements.txt",
    "project_summary.txt",
    "cookies.txt",
    
    # Uploads and data
    "OVERVIEW.md",
    "TODO.md",
    "gpt",
    "uploads",
    "logs",
    "transcripts",
    "scripts",
    "tests",
    "chatgpt_conversations",
    "to_txt.py",
]

# Output file
output_file = r"scripts\project_summary.txt"

def should_ignore(path):
    for ignore in ignore_list:
        if ignore in path:
            return True
    return False

def main():
    dir_ = input("Enter the directory to scan: ")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(dir_):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d))]
            for file in files:
                file_path = os.path.join(root, file)
                if not should_ignore(file_path):
                    outfile.write(f"--- {file_path} ---\n")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except UnicodeDecodeError as e:
                        print(f"Error reading {file_path}: {e}")
                    outfile.write("\n\n")

if __name__ == "__main__":
    main()