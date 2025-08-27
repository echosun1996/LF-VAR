import os
import glob
import shutil

select_name = "ISIC_0031276.jpg"
select_folder = "vasc"

output_folder = "./output"
infection_list = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

select_seg_name = select_name.replace(".jpg", "_segmentation.png")
img_folder = '/mnt/SkinGenerativeModel/code/data/local/HAM10000/input/val/HAM10000_img_class/' + select_folder.lower()
seg_folder = '/mnt/SkinGenerativeModel/code/data/local/HAM10000/input/val/HAM10000_seg_class/' + select_folder.lower()
gen_root = '/mnt/SkinGenerativeModel/code/data/compare_results/main_cross_infer'

if os.path.exists(os.path.join(output_folder, select_folder.upper())):
    shutil.rmtree(os.path.join(output_folder, select_folder.upper()))

if not os.path.exists(os.path.join(img_folder, select_name)):
    print(os.path.join(img_folder, select_name), "not exist!")
    exit(-1)
else:
    for inf in infection_list:
        gen_path = os.path.join(gen_root, select_folder.lower() + "_" + inf.lower())
        output_subfolder = os.path.join(output_folder, select_folder.upper(), inf.upper())

        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        pattern = os.path.join(gen_path, select_name.replace(".jpg", "_*") + ".png")
        matched_files = sorted(glob.glob(pattern))

        if matched_files:
            print(f"Found {len(matched_files)} files in {gen_path} for {select_name}.")
            top_files = matched_files[:3]
            for file in top_files:
                output_file = os.path.join(output_subfolder, os.path.basename(file))
                shutil.copy(file, output_file)
                print(f"Copied {file} to {output_file}")
        else:
            print(f"No files found in {gen_path} for {select_name}.")

    final_output_folder = os.path.join(output_folder, select_folder.upper())
    if not os.path.exists(final_output_folder):
        print(final_output_folder + " not exist!")
        exit(-1)
    src_img_file = os.path.join(img_folder, select_name)
    dst_img_file = os.path.join(final_output_folder, select_name)
    if os.path.exists(src_img_file):
        shutil.copy(src_img_file, dst_img_file)
        print(f"Copied {src_img_file} to {dst_img_file}")
    else:
        print(f"{src_img_file} not found!")

    src_seg_file = os.path.join(seg_folder, select_seg_name)
    dst_seg_file = os.path.join(final_output_folder, select_seg_name)
    if os.path.exists(src_seg_file):
        shutil.copy(src_seg_file, dst_seg_file)
        print(f"Copied {src_seg_file} to {dst_seg_file}")
    else:
        print(f"{src_seg_file} not found!")