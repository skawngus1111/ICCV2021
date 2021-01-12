import numpy as np
import matplotlib.pyplot as plt

class RandSectorReject() :
    def __init__(self, image_size, theta_dist, start_radius, sector_length, num_mask,
                 maximum_angle, ring_length):
        self.image_size = (image_size, image_size, 3)
        self.theta_dist_ = theta_dist
        self.start_radius = start_radius

        self.num_mask = num_mask
        self.theta = 0
        self.sector_angle = maximum_angle
        self.sector_length = sector_length

        self.ring_angle = 360
        self.ring_length = ring_length

        self.angle = 0
        self.radius = 0
        self.count = 5

    def __call__(self, img):
        reject_img = self.reject_frequency(img)

        return reject_img

    def get_sector_spec(self):
        self.angle = self.sector_angle
        self.length = self.sector_length
        self.radius = self.start_radius  # choose radius of thin ring shape

    def get_mask(self):
        mask = np.ones((int(self.image_size[0]), int(self.image_size[0])))

        if self.theta_dist_ :
            self.theta = np.random.choice(360, p=self.theta_dist)  # choose starting angle of sector shape
        else :
            self.theta = np.random.randint(0, 359) # choose starting angle of sector shape

        for theta in [self.theta, self.theta + 180] :
            radius_range = np.arange(self.radius, self.radius + self.length, 0.1).reshape(-1, 1)
            theta_range = np.arange(theta - self.angle//2, theta + self.angle//2+1, 0.1).reshape(-1, 1)

            x_unit = np.cos(np.radians(theta_range))
            y_unit = np.sin(np.radians(theta_range))

            x = np.array(np.multiply(radius_range.T, x_unit), dtype=np.int16) + self.image_size[0] // 2
            y = np.array(np.multiply(radius_range.T, y_unit), dtype=np.int16) + self.image_size[1] // 2

            ## 3. Get mask from rectangle coordinates
            mask[x, y] = 0

        return mask

    def get_theta_dist(self, img_f):

        pixel_sum_ver_theta = []

        theta = np.arange(0, 360).reshape(-1, 1)

        radius = np.arange(0, self.image_size[0]//2).reshape(-1, 1)

        x_unit = np.cos(np.radians(theta))
        y_unit = np.sin(np.radians(theta))

        x = np.array(np.multiply(radius.T, x_unit), dtype=np.int16) + self.image_size[0] // 2
        y = np.array(np.multiply(radius.T, y_unit), dtype=np.int16) + self.image_size[1] // 2
        for i in range(x.shape[0]) :
            pixel_sum_ver_theta.append(np.abs(np.sum(img_f[x[i], y[i]])))

        self.theta_dist = pixel_sum_ver_theta / np.sum(pixel_sum_ver_theta)

    def reject_frequency(self, img):
        reject_img_3c = np.zeros_like(img.reshape((256, 256, -1)))
        img_f = np.fft.fft2(img[:, :, 0].reshape((self.image_size[0], self.image_size[1])))
        img_f = np.fft.fftshift(img_f)
        spectrum = np.log(np.abs(img_f))
        masking_img_f = img_f
        reject_spectrum = spectrum

        for _ in range(self.num_mask) :
            self.get_theta_dist(img_f)
            mask = self.get_mask()
            masking_img_f = masking_img_f * mask
            reject_spectrum = reject_spectrum * mask

        img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(masking_img_f)))
        reject_img_3c[:, :, 0] = img_back
        reject_img_3c[:, :, 1] = img_back
        reject_img_3c[:, :, 2] = img_back

        return reject_img_3c

def plot_example(data, save_name, count) :
    if 'spectrum' in save_name :
        height, width = data.shape
        figsize = (1, height / width) if height >= width else (width / height, 1)
        plt.figure(figsize=figsize)

        plt.imshow(data, cmap='gray')
    else :
        height, width, _ = data.shape
        figsize = (1, height / width) if height >= width else (width / height, 1)
        plt.figure(figsize=figsize)

        plt.imshow(data)

    plt.xticks([]); plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)

    try :
        plt.savefig('{}{}.png'.format(save_name, count), dpi=300)
    except FileNotFoundError :
        import os
        os.makedirs(save_name)
        plt.savefig('{}{}.png'.format(save_name, count), dpi=300)
    plt.close()