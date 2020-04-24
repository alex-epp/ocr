from utils.iam import IAMWords, Resize

if __name__ == '__main__':
    # ROOT = r'D:\Alex\datasets\iamDB'
    ROOT = r'C:\datasets\iamDB'

    ds = IAMWords(ROOT,
                  split='train',
                  transform=Resize(256, 32))

    import matplotlib.pyplot as plt
    for image, label in ds:
        plt.imshow(image)
        plt.title(label)
        plt.show()
