
python src/cleanup_dataset.py data/training.txt --output=data/cleaned_data.txt --thread=20
python src/cleanup_dataset.py data/validation.txt --output=data/cleaned_validation.txt --thread=20 --cleanup_only
