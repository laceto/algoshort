import os

def print_folder_structure(root_path, indent=""):
    try:
        items = sorted(os.listdir(root_path))
    except PermissionError:
        print(indent + "🚫 [Permission Denied]")
        return

    for item in items:
        item_path = os.path.join(root_path, item)
        if os.path.isdir(item_path):
            print(f"{indent}📁 {item}/")
            print_folder_structure(item_path, indent + "    ")
        else:
            print(f"{indent}📄 {item}")

# 🔍 Usage
import os
cwd = os.getcwd()
print(cwd)

# if __name__ == "__main__":
#     project_path = "algoshort/src"  # Replace with your actual folder path
#     # project_path = cwd
#     print(f"📦 Project Structure: {project_path}")
#     print_folder_structure(project_path)
