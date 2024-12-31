import kagglehub

kagglehub.login()

# Download latest version
path = kagglehub.model_download("google/gemma-2/transformers/gemma-2-2b")

print("Path to model files:", path)