python src/feature_transformation.py data/cleaned_data.txt --output=training
python src/feature_transformation.py data/cleaned_validation.txt --fit

python src/feature_transformation.py data/cleaned_data.txt --output=training --hin
python src/feature_transformation.py data/cleaned_validation.txt --fit --hin
