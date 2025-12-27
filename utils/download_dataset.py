from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="NONHUMAN-RESEARCH/TEST_RECORD_ANNOTATIONS",
    repo_type="dataset",
    cache_dir="/home/jorge/ds"
)

print("Downloaded to:", path)
