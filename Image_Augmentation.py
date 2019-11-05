
import random
import numpy as np
from keras.preprocessing.image import array_to_img

def Random_Erasing(img):

    # ================ 各パラメータ ================ 
    H, W, cl = img.shape    # 画像の幅（W）と高さ（H）を得る
    S = W*H # 入力画像の面積
    p = 0.5 # Random Erasingを適用するかの閾値
    s_l = 0.02  # Erasing領域の比率の範囲
    s_h = 0.4   # Erasing領域の比率の範囲
    r1 = 0.3    # Erasingアスペクト比の範囲
    r2 = 3   # Erasingアスペクト比の範囲
    
    
    # 確率の初期化
    p_init = random.random()
    
    if p_init >= p:
        out = img
    else:
        while True:
            S_e = random.uniform(s_l, s_h)*S
            r_e = random.uniform(r1, r2)
            H_e = int(np.sqrt(S_e*r_e))
            W_e = int(np.sqrt(S_e/r_e))
            x_e = random.randint(0, H)
            y_e = random.randint(0, W)
        
            if y_e + W_e <= W and x_e + H_e <= H:
                break
    
        img[x_e:(x_e + H_e), y_e:(y_e + W_e), :] = np.random.randint(0, 255, (H_e, W_e, 3))
        out = img
        
    return out


def process_imgs(generator, img, dir_name, path, num):    
    
    g = generator.flow(img, batch_size=1)

    # 入力画像から拡張する画像枚数を指定
    for i in range(num):
        bach = g.next()    
        out = Random_Erasing(bach[0])
        out_img = array_to_img(out)
        out_img.save(dir_name + '/' + path + str(i) + '.jpg', quality=95)       
        

