import kagglehub

# Download latest version
path = kagglehub.dataset_download("koryakinp/fingers")

print("Path to dataset files:", path)
