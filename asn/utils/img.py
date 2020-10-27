import cv2
import numpy as np


def flip_img(img, rgb_to_front=True):
    """ changing image shape (w,h,3)->(3,w,h) """
    if rgb_to_front:
        return np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
    else:
        return img.transpose((1, 2, 0))


def flip_imgs(imgs, rgb_to_front=True):
    """ changing images shapes (n,w,h,3)->(n,3,w,h) """
    if rgb_to_front:
        return np.swapaxes(np.swapaxes(imgs, 2, 3), 1, 2)
    else:
        return np.swapaxes(np.swapaxes(imgs, 1, 2), 2, 3)


def np_rgb_to_cv(np_array_rgb):
    """ flip rgb to bgr """
    return np_array_rgb[..., ::-1]


def np_shape_to_cv(shape):
    """flip to match cv
        Mat (rows, cols)<=>(height, width)
        but size in cv (width, height)
    """
    return shape[::-1]


def convert_to_uint8(image_in):
    temp_image = np.float32(np.copy(image_in))
    cv2.normalize(temp_image, temp_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)

    return temp_image.astype(np.uint8)


def torch_tensor_to_img(t):
    s = t.size()
    if len(s) == 3:
        # one img
        return flip_img(t.data.cpu().numpy(), rgb_to_front=False)
    elif len(s) == 4:
        return flip_imgs(t.data.cpu().numpy(), rgb_to_front=False)
    else:
        raise ValueError("wrong tensor shape to convert to images")


def add_gauss_noise(image, mean=0, var=0.001):
    row, col, ch = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy


def resize_with_border(img, shape, border_color=[1, 1, 1]):
    border_v = 0
    border_h = 0
    im_col = shape[0]
    im_row = shape[1]
    if (im_col / im_row) >= (img.shape[0] / img.shape[1]):
        border_v = int((((im_col / im_row) * img.shape[1]) - img.shape[0]) / 2)
    else:
        border_h = int((((im_row / im_col) * img.shape[0]) - img.shape[1]) / 2)
    img = cv2.copyMakeBorder(img, border_v, border_v, border_h, border_h,
                             cv2.BORDER_CONSTANT, value=border_color)
    return cv2.resize(img, (im_row, im_col))


def _fill_empty_imgs(imgs, color):
    """fills a 2d img array so that all shapes are same"""
    assert len(imgs) and len(imgs[0]), "no images"
    max_len = sorted([len(i_v) for i_v in imgs], reverse=True)[0]
    fill_img = np.full(imgs[0][0].shape, color)

    def add_fillers(imgs_to_fill):
        # for _ in range(len(imgs_to_fill) - max_len):
        n = max_len - len(imgs_to_fill)
        if n:
            n_filler = np.full((n, *imgs_to_fill[0].shape), color)
            imgs_to_fill = np.concatenate((imgs_to_fill, n_filler))
        return imgs_to_fill

    return [add_fillers(i_v) for i_v in imgs]


def montage(imgs=[[]], margin_top=20, margin_bottom=20,
            margin_left=20, margin_right=20,
            margin_separate_vertical=20, margin_separate_horizontal=20,
            margin_color_bgr=[1, 1, 1],
            titles=None, fontScale=0.5, title_pos=None):
    """ creates a montage image with imgs in a grid layout
    Args:
        imgs: 2d list with numpy images with same shapes
        titles: 2d list with strings used das a title at the top of a img,
                must have same shape as imgs
        title_pos: position in img with margins

    Returns:
        The return a nparray with (n_task_frames,k,((k_index_vid,k_distance,k_frame_index)
    usage:
        black = np.zeros((300,300,3))
        white= np.ones((300,300,3))
        titles = [["black","white","black"],
                  ["white","black","white"],
                  ["black","white","black"]]
        imgs =[[black,white,black],[white,black,white],[black,white,black]]
        imgs_concat=montage(imgs,titles=titles,margin_color_bgr=[0.5,0.5,0.5])
        cv2.imshow('Main', imgs_concat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        :param margin_top:
        :param margin_bottom:
        :param margin_left:
        :param margin_right:
        :param margin_separate_vertical:
        :param margin_separate_horizontal:
        :param margin_color_bgr:
        :param fontScale:
    """
    imgs_border = []
    imgs = _fill_empty_imgs(imgs, margin_color_bgr)
    # add text and borders
    for v_index, i_v in enumerate(imgs):
        h_imgs = []
        for h_index, i_h in enumerate(i_v):
            # add borders
            # margin sep. is used for between imgs
            m_top = margin_separate_horizontal if v_index != 0 else margin_top
            m_bot = 0 if v_index != len(imgs) - 1 else margin_bottom
            m_l = margin_separate_vertical if h_index != 0 else margin_left
            m_r = 0 if h_index != len(i_v) - 1 else margin_right

            i = cv2.copyMakeBorder(i_h, m_top, m_bot, m_l, m_r,
                                   cv2.BORDER_CONSTANT, value=margin_color_bgr)
            if titles is not None:
                text = ""
                if v_index < len(titles) and h_index < len(titles[v_index]):
                    text = titles[v_index][h_index]
                title_font = cv2.FONT_HERSHEY_SIMPLEX
                fontColor = (0, 0, 0)  # black
                thickness = 1
                text_size, _ = cv2.getTextSize(
                    text, title_font, fontScale, thickness)
                # title position at the top of img
                if title_pos is None:
                    title_pos_top = (m_l, max(
                        m_top - text_size[1], text_size[1]))
                cv2.putText(i, text, title_pos_top,
                            title_font, fontScale, fontColor, thickness)
            h_imgs.append(i)
        imgs_border.append(h_imgs)
    # create one image
    v_horizontal_con = [np.concatenate(im, axis=1) for im in imgs_border]
    imgs_concat = np.concatenate(v_horizontal_con, axis=0)
    return imgs_concat


if __name__ == '__main__':
    img_size = (300, 200)
    black = np.zeros((*img_size, 3))
    white = np.ones((*img_size, 3))
    titels = [["black", "white", "black"],
              ["white", "black-rot", "white"],
              ["black", "white", "black"]]
    black_rot = cv2.rotate(black, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    black_rot = resize_with_border(
        black_rot, img_size, border_color=[0.5, 0.5, 0.5])
    imgs = [[black, white, black], [
        white, black_rot, white], [black, white, black]]
    imgs_concat = montage(imgs, titles=titels,
                          margin_color_bgr=[0.5, 0.5, 0.5])
    cv2.imshow('Main', imgs_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # create a vid
    fourcc = cv2.VideoWriter_fourcc('j', 'p', 'e', 'g')
    shape = np_shape_to_cv(imgs_concat.shape[:2])
    vid_writer = cv2.VideoWriter("out.mov", fourcc, 1, shape)
    for frame in range(10):
        bgr = np.uint8(np_rgb_to_cv(imgs_concat) * 255)
        cv2.imwrite("tmp.jpg", bgr)
        imgs = np.roll(np.roll(imgs, 1, axis=1), 1, axis=0)
        imgs_concat = montage(imgs, titles=titels,
                              margin_color_bgr=[0.5, 0.5, 0.5])
        vid_writer.write(cv2.resize(bgr, shape, cv2.INTER_NEAREST))
    vid_writer.release()
