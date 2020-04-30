import argparse
import os
from PIL import Image


def resize_image(image, size):                     #resize一张图片
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)     #resize指定大小和质量，Image.ANTIALIAS高质量

def resize_images(image_dir, output_dir, size):    #批量resize图片后保存到output_dir
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):    #os操作系统，os.path.exists判断括号里的文件是否存在，（文件路径）
        os.makedirs(output_dir)           #os.makedirs(path, mode=0o777)递归创建目录，path - 需要递归创建的目录，mode - 权限模式

    images = os.listdir(image_dir)        #os.listdir(path)返回指定的文件夹包含的文件或文件夹的名字的列表，path - 需要列出的目录路径
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:     #os.path.join连接两个或更多的路径名组件
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)    #.save保存，format？？？
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))

def main(args):                                 #args：参数
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]
    resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()   #命令行工具argparse，创建解析器，创建一个 ArgumentParser 对象
    parser.add_argument('--image_dir', type=str, default='./data/train2014/',     #add_argument:读入命令行参数，
                        help='directory for train images') 
    parser.add_argument('--output_dir', type=str, default='./data/resized2014/',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()           #parse_args(args=None, nampespace=None)将之前add_argument()定义的参数args进行赋值namespace，并返回namespace
    main(args)
