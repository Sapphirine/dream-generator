import shutil, os


src = 'models'
dest = 'extern\pytorch-CycleGAN-and-pix2pix\checkpoints'

# create the destination directory
if not os.path.exists(dest):
    os.mkdir(dest)

# copy the models to the checkpoints folder
folders = os.listdir(src)
for folder in folders:
    # create a new folder at the destination directory
    dest_folder_path = os.path.join(dest, folder)
    if not os.path.exists(dest_folder_path):
        os.mkdir(dest_folder_path)

    # copy the model to the destination folder
    src_file_path = os.path.join(src, folder)
    for file in os.listdir(os.path.join(src, folder)):
        shutil.copy(os.path.join(src_file_path, file), dest_folder_path)
        print(file + " copied to " + dest_folder_path)
