import tempfile
import subprocess
import os
import glob 

from PIL import Image

def run_semantic_segmentation(imgs: list, index_digits: int):

    # Create a temporary folder 
    temp_dir = tempfile.TemporaryDirectory()
    
    # Save the target resolutions
    img_res = []
    for file in imgs:
        img = Image.open(file)
        w,h = img.size[0], img.size[1]
        img_res.append((w,h))

        # Resize if the image is to large
        if w > 1920 or h > 1280:
            img = img.resize((w//2,h//2), Image.ANTIALIAS)
        img.save(os.path.join(temp_dir.name, file.split(os.sep)[-1]))

    args =  f'--dataset cityscapes --cv 0 --fp16 --bs_val 1 --eval folder ' \
            '--eval_folder {} --n_scales 0.5,1.0,2.0 '\
            '--snapshot src/py/deps/semantic_segmentation/pretrained_models/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth '\
            '--arch ocrnet.HRNet_Mscale --result_dir {}'.format(temp_dir.name, os.path.join(temp_dir.name,'semantic_seg'))

    # Run the semantic segmentation
    cmd = 'external/semantic-segmentation/train ' + args
    subprocess.Popen(cmd, shell=True).wait()

    predictions = sorted(glob.glob(os.path.join(temp_dir.name,'semantic_seg','best_images', f"{'?'*index_digits}_prediction.png")))

    assert len(predictions) == len(img_res), "Number of semantic segmentation predictions is not the same as the number of input images"

    for (idx, pred_img), input_image in zip(enumerate(predictions), imgs):
        img_name = os.path.basename(input_image).split('.')[0]

        img = Image.open(pred_img)
        w,h = img.size[0], img.size[1]
        if w != img_res[idx][0] or h != img_res[idx][1]:
            img = img.resize(img_res[idx], Image.ANTIALIAS)

        img.save(input_image.replace(img_name, f"{img_name}_sem").replace('.jpg','.png').replace('.jpeg','.png'))

    # Delete the temporary folder
    temp_dir.cleanup()