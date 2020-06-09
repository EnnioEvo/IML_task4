1) Run generate_validation.py in order to generate val_triplets.txt
	Note: this step can be avoided setting submit=True in main_features.py and commenting the lines depending on val_triplets.txt
2) Run generate_features.py
	Note: this step takes significantly less using a GPU
3) Run main.py