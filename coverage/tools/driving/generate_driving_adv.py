import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random


def decode_image(x):
    x = x.reshape((100, 100, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def add_light(temp, gradients):
    # import skimage
    temp = temp.reshape(temp.shape[0], -1)
    gradients = gradients.reshape(gradients.shape[0], -1)
    new_grads = np.ones_like(gradients)
    grad_mean = 500 * np.mean(gradients, axis=1)
    grad_mean = np.tile(grad_mean, temp.shape[1])
    grad_mean = grad_mean.reshape(temp.shape)
    temp = temp + 10 * new_grads * grad_mean
    temp = temp.reshape(temp.shape[0], 100, 100, 3)
    return temp

def add_black(temp, gradients):
    rect_shape = (10, 10)
    for i in range(temp.shape[0]):
        orig = temp[i].reshape(1, 100, 100, 3)
        grad = gradients[i].reshape(1, 100, 100, 3)
        start_point = (
            random.randint(0, grad.shape[1] - rect_shape[0]), random.randint(0, grad.shape[2] - rect_shape[1]))
        new_grads = np.zeros_like(grad)
        patch = grad[:, start_point[0]:start_point[
            0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
                      start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
        orig = orig + 100 * new_grads
        temp[i] = orig.reshape(100, 100, 3)
    return temp


black_adv = r"C:\Users\yanmi\Desktop\driving_dave-orig_black_adv_x.npy"
light_adv = r"C:\Users\yanmi\Desktop\driving_dave-orig_light_adv_x.npy"
clean_adv = r"C:\Users\yanmi\Desktop\driving_dave-orig_clean_x.npy"
clean_y_path = r"C:\Users\yanmi\Desktop\driving_dave-orig_clean_y.npy"

clean_x = np.load(clean_adv)
clean_y = np.load(clean_y_path)
# light_x = np.load(light_adv)
# black_x = np.load(black_adv)

# idx = 200

# # img = decode_image(clean_x[idx].copy())
# # plt.imshow(img)
# # plt.show()
# # temp = clean_x.copy()
# # pert = 1 * np.random.normal(size=clean_x.shape)
# # temp = add_light(temp, pert)
# # img = decode_image(temp[idx])
# # plt.imshow(img)
# # plt.show()

# # # save 1 image
# # img = decode_image(clean_x[idx])
# # # imsave('orig.png', img)
# # plt.imshow(img)
# # plt.axis('on') # 关掉坐标轴为 off
# # plt.title('image') # 图像题目
# # plt.show()

# # # add perturbation
# # temp = clean_x.copy()
# # pert = 1 * np.random.normal(size=temp.shape)
# # for i in range(5):
# #     temp = add_black(temp, pert)
# # img = decode_image(temp[idx])
# # # imsave('pert.png', img)
# # plt.imshow(img)
# # plt.axis('on') # 关掉坐标轴为 off
# # plt.title('image') # 图像题目
# # plt.show()




temp1 = clean_x.copy()
pert1 = 1 * np.random.normal(size=clean_x.shape)
for i in range(5):
    temp1 = add_black(temp1, pert1)
black_x = temp1.copy().astype("float32")
print(type(black_x))



temp2 = clean_x.copy()
pert2 = 1 * np.random.normal(size=clean_x.shape)
temp2 = add_light(temp2, pert2)
light_x = temp2.copy().astype("float32")



print(clean_x.shape,np.min(clean_x), np.max(clean_x))
print(light_x.shape,np.min(light_x), np.max(light_x))
print(black_x.shape,np.min(black_x), np.max(black_x))

np.savez("driving_dave-orig_nature.npz",inputs=clean_x,labels=clean_y.copy())
np.savez("driving_dave-orig_light.npz",inputs=light_x,labels=clean_y.copy())
np.savez("driving_dave-orig_black.npz",inputs=black_x,labels=clean_y.copy())
print(os.getcwd())
clean_x = np.load("driving_dave-orig_nature.npz",allow_pickle=True)
light_x = np.load("driving_dave-orig_light.npz",allow_pickle=True)
black_x = np.load("driving_dave-orig_black.npz",allow_pickle=True)

data_dict = {"nature":clean_x,"light":light_x,"black":black_x}
idx = 200
img_idx = 1
for attack in ["nature", "light", "black"]:
    data = data_dict[attack]["inputs"]
    print(data.dtype)
    labels = data_dict[attack]["labels"]
    indices = [0,100,200,300,400]
    for j,idx in enumerate(indices):
        img = data[idx]
        img = img.squeeze()
        print(img.shape,img.min(),img.max())
        img = decode_image(img)
        plt.subplot(3,len(indices),img_idx)
        img_idx += 1
        plt.title(f"{attack}_{j},{labels[idx]}")
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
plt.show()
