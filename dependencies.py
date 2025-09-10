import os
import ast
import importlib.util
import sys

# Optional: fallback mapping for common packages
fallback_mapping = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "Crypto": "pycryptodome",
}

def collect_imports_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        node = ast.parse(f.read(), filename=file_path)
    imports = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Import):
            for alias in item.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(item, ast.ImportFrom):
            if item.module:
                imports.add(item.module.split('.')[0])
    return imports

def collect_all_imports(root_dir):
    all_imports = set()
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                file_imports = collect_imports_from_file(file_path)
                all_imports.update(file_imports)
    return sorted(all_imports)

def map_to_pypi(imports):
    dependencies = set()
    for imp in imports:
        # Try to resolve via importlib
        if importlib.util.find_spec(imp):
            dependencies.add(imp)
        elif imp in fallback_mapping:
            dependencies.add(fallback_mapping[imp])
        else:
            print(f"⚠️ Unknown or local module: {imp}")
    return sorted(dependencies)

# 🔍 Usage
if __name__ == "__main__":
    project_dir = "src"  # Change this to your project folder
    raw_imports = collect_all_imports(project_dir)
    dependencies = map_to_pypi(raw_imports)

    print("\n📦 Dependencies for pyproject.toml:")
    for dep in dependencies:
        print(f'"{dep}",')
