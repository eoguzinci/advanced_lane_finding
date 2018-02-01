import os

def rename_files():
	file_list = os.listdir("./recamera/")
	saved_path = os.getcwd()
	os.chdir("./recamera/")

	for idx, file_name in enumerate(file_list):
		if idx == 9:
			break
		num = file_name[11:]
		os.rename(file_name, file_name[:11]+"0"+num)

rename_files()