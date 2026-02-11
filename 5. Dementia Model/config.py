output_dir = "diagnosis"
modalities = ["wav", "specto", "trans"]  # source folders
original_classes = ["control", "dementia"]  # original folder names
class_map = {"control": "cn", "dementia": "ad"}  # target folder names
test_csv_name = "task1.csv"