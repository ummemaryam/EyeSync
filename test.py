import os

def print_directory_structure(start_path, indent="", depth=2):
    if depth < 1:
        return  # Stop recursion if depth limit is reached

    for item in os.listdir(start_path):
        item_path = os.path.join(start_path, item)
        if os.path.isdir(item_path):
            print(f"{indent}[DIR] {item}")
            print_directory_structure(item_path, indent + "  ", depth - 1)
        else:
            print(f"{indent}{item}")

if __name__ == "__main__":
    project_directory = os.getcwd()  # Current directory
    print(f"Directory structure of: {project_directory}")
    print_directory_structure(project_directory, depth=2)
