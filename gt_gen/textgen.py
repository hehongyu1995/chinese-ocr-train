# coding:utf-8
##绘制图像
from PIL import Image, ImageDraw, ImageFont
import glob
import random
import sys
import numpy as np
from  uuid import uuid1
import os
from custom_charset import markReplaceDict
from keys import alphabet as charset
import time
from random_eraser import get_random_eraser

erase = get_random_eraser(p=1, s_l=0.01, s_h=0.02, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, pixel_level=False)


def getRandomDateList(num=20):
    a1 = (2000, 1, 1, 0, 0, 0, 0, 0, 0)  # 设置开始日期时间元组（2000-01-01 00：00：00）
    a2 = (2050, 12, 30, 23, 59, 59, 0, 0, 0)  # 设置结束日期时间元组（2050-12-31 23：59：59）

    start = time.mktime(a1)  # 生成开始时间戳
    end = time.mktime(a2)  # 生成结束时间戳
    ret = []
    # 随机生成10个日期字符串
    for i in range(num):
        t = random.randint(start, end)  # 在开始和结束时间戳中随机取出一个
        date_touple = time.localtime(t)  # 将时间戳生成时间元组
        date = time.strftime(u"%Y-%m-%d", date_touple)  # 将时间元组转成格式化字符串（1976-05-21)
        date = date.split(u'-')
        date = u'{}年{}月{}日'.format(date[0], date[1], date[2])
        ret.append(date)
    return ret


def getRandomId(num=20, length=10):
    chars = [u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'I', u'I', u'I', u'I', u'J', u'J', u'J', u'J', u'J',
             u'K', u'L', u'M', u'N',
             u'O', u'P', u'Q', u'R', u'S', u'T', u'U', u'V', u'W', u'W', u'W', u'W', u'W', u'X', u'Y', u'Z', u'1', u'1',
             u'1', u'2', u'3',
             u'4', u'5', u'6', u'7', u'8', u'9', u'0', u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k',
             u'l', u'l', u'l', u'l', u'm', u'n',
             u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y', u'z']
##remove IOZSV and lower case
#    chars = [u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'J',
#             u'K', u'L', u'M', u'N',
#             u'P', u'Q', u'R', u'S', u'T', u'U', u'W', u'X', u'Y',
#             u'1', u'2', u'3',
#             u'4', u'5', u'6', u'7', u'8', u'9', u'0']
    ret = []
    for i in range(num):
        id = u''
        for j in range(length):
            id += random.choice(chars)
        ret.append(id)
    return ret


def rotate_box_tramform(lineboxes, center, angleTuple=None):
    """
    @@lineboxes:box集合
    @@center :图像旋转中心点
    @@angleTuple:旋转角度范围:(min,max)
    按指定的角度旋转
    """
    if angleTuple is None:
        angleTuple = (-10, 10)

    angle = random.uniform(angleTuple[0], angleTuple[1])  ##随机获取一个角度

    lineBoxes = []
    for linebox in lineboxes:
        lineBoxes.append([])
        for box in linebox:
            box = rotate_box(angle, box, center)
            lineBoxes[-1].append(box)
    return lineBoxes, angle

    # return [rotate_box(angle,box,center)  for box in boxes],angle


def rotate_box(angle, box, center):
    """
    @@angle:旋转角度
    @@box:旋转前的box
    @@center:旋转中心点
    旋转图像，同样文本box也随着旋转，返回取回后的图像及box
    x0= (x - rx0)*cos(a) - (y - ry0)*sin(a) + rx0 ;

    y0= (x - rx0)*sin(a) + (y - ry0)*cos(a) + ry0 ;
    """
    xmin, ymin, xmax, ymax = box
    cX, cY = center
    angle = -angle / 180.0 * np.pi
    xmin_ = (xmin - cX) * np.cos(angle) - (ymin - cY) * np.sin(angle) + cX
    ymin_ = (xmin - cX) * np.sin(angle) + (ymin - cY) * np.cos(angle) + cY

    xmax_ = (xmax - cX) * np.cos(angle) - (ymax - cY) * np.sin(angle) + cX
    ymax_ = (xmax - cX) * np.sin(angle) + (ymax - cY) * np.cos(angle) + cY
    # xmin_ = xmax_ - (xmax-xmin)
    # ymin_ = ymax_ - (ymax-ymin)
    # print xmax-xmin,xmax_-xmin_,ymax-ymin,ymax_-ymin_
    return int(xmin_), int(ymin_), int(xmax_), int(ymax_)


def draw_underlined_text(draw, pos, text, font, **options):
    twidth, theight = draw.textsize(text, font=font)
    lx, ly = pos[0], pos[1] + theight
    draw.text(pos, text, font=font, **options)
    draw.line((lx, ly, lx + twidth, ly), width=2, **options)


def draw_box(labels, size=(512, 512), im=None):
    """
    绘制文字
    @@labels：文本集
    @@size:图像的大小
    @@im:如果im为None,需传入背景图像，否则Image.new生成一张图像
    """
    boxes = []
    lineBoxes = []
    lineChars = []
    chars = []
    X, Y = size
    x, y = 0, 0

    initX, initY = int(size[0] * 0.1), int(size[0] * 0.1)
    cX = initX
    cY = initY
    lineMaxY = 0  ##行最大值
    if im is None:
        im = Image.new(mode='RGB', size=(X, Y), color='white')  # color 背景颜色，size 图片大小
    drawer = ImageDraw.Draw(im)
    fontType = random.choice(fonts)  ##随机获取一种字体

    isDraw = True
    tmpImg = np.array(im)[cY:-cY, cX:-cX]
    # fill0,fill1,fill2 = tmpImg[:,:,0].mean(),tmpImg[:,:,1].mean(),tmpImg[:,:,2].mean()
    # fill = random.randint(0,255)
    fillmean = int(tmpImg.mean())
    #print('fill mean', fillmean)
    if fillmean < 80:
        fill = random.randint(fillmean, 255)
    else:
        fill = random.randint(0, 120)
    fill = (fill, fill, fill)
    # fill = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
    # fill = (random.randint(0,int(fill0)),random.randint(0,int(fill1)),random.randint(0,int(fill2)))
    #print("label_len", len(labels))
    for label in labels:
        #print(label)
        fontSize = random.randint(20, 50)  # 字体大小
        font = ImageFont.truetype(fontType, fontSize)
        textSize = drawer.textsize(label, font=font)
        lineBox = []
        lineChar = []
        lineNum = 0
        for char in label:
            if markReplaceDict.has_key(char):
                #print(u'Replace SBC case {} with {} .'.format(char, markReplaceDict[char]))
                char = markReplaceDict[char]
            if charset.find(char) == -1:
                #print(u'Not found {} in charset. skip it. '.format(char))
                continue
            charX, charY = drawer.textsize(char, font=font)  ##字符所占的宽度
            if charY + cY < Y - initY:
                if charX + cX < X - initX and (lineNum < maxLen or maxLen is None):  ##判断当前字符能否在此行中放下
                    boxes.append([cX, cY, cX + charX, cY + charY])
                    lineBox.append([cX, cY, cX + charX, cY + charY])

                    drawer.text(xy=(cX, cY), text=char, font=font, fill=fill)
                    draw_underlined_text(draw=drawer, pos=(cX, cY), text=char, font=font, fill=fill)
                    chars.append(char)
                    lineChar.append(char)

                    cX = cX + charX
                    if lineMaxY < charY:
                        lineMaxY = charY
                    lineNum += 1

                else:
                    ##将未能放下的字符移动至下一行
                    lineNum = 0
                    lineBoxes.append(lineBox)
                    lineChars.append(lineChar)
                    lineBox = []
                    lineChar = []
                    cX, cY = initX, cY + lineMaxY + np.random.randint(0, 10)
                    if cY + charY < Y - initY:
                        drawer.text(xy=(cX, cY), text=char, font=font, fill=fill)
                        draw_underlined_text(draw=drawer, pos=(cX, cY), text=char, font=font, fill=fill)
                        lineMaxY = charY
                        boxes.append([cX, cY, cX + charX, cY + charY])
                        lineBox.append([cX, cY, cX + charX, cY + charY])
                        chars.append(char)
                        lineChar.append(char)
                        cX = cX + charX
                    else:
                        isDraw = False
                        break
            else:
                isDraw = False
                break
        lineNum = 0
        cX, cY = initX, cY + lineMaxY + np.random.randint(0, 10)
        lineBoxes.append(lineBox)
        lineChars.append(lineChar)
        lineBox = []
        lineChar = []

        if not isDraw:
            break
    return im, boxes, chars, lineBoxes, lineChars


import cv2
import numpy as np


def rectangle(img, boxes):
    tmp = np.copy(img)
    # tmp = np.zeros(img.shape,dtype=np.uint8)
    for box in boxes:
        cv2.rectangle(tmp, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 255))
    return tmp


import numpy as np


def read_text(p=None):
    """
    获取语料文本数据
    """
    IntP = 10  ##默认取五个文件然后随取抽取一部分文本
    dataList = []
    for i in range(IntP):

        if p is None:
            p = random.choice(corpusPaths)
        with open(p) as f:
            data = f.read().decode('utf-8')
        data = [line.strip() for line in data.split(u'\n') if line.strip() != u'' and len(line.strip()) > 1]
        for line in data:
            if len(line) > maxLen:
                sub_string = [substr + u"," for substr in line.split(u'，')]
                dataList.extend(sub_string)
                sub_string = [substr + u"、" for substr in line.split(u'、')]
                dataList.extend(sub_string)
        dataList.extend(data)
    dataList.extend(getRandomDateList(num=80))
    dataList.extend(getRandomId(num=80))
    np.random.shuffle(dataList)
    return np.array(dataList)


def read_text_split(p=None, length=4, lineLength=300):
    """
    获取语料文本数据
    按照10个字一行分隔
    """
    IntP = 30  ##默认取五个文件然后随取抽取一部分文本
    dataList = []
    for i in range(IntP):

        if p is None:
            p = random.choice(corpusPaths)
        with open(p) as f:
            data = f.read().decode('utf-8')
        data = [line.strip() for line in data.split(u'\n') if line.strip() != u'' and len(line.strip()) > 1]

        dataList.extend(data)
    np.random.shuffle(dataList)
    splitPatters = [u',', u':', u'-', u' ', u';', u'。']
    splitPatter = np.random.choice(splitPatters, 1)
    data = splitPatter[0].join(dataList)
    splitData = []
    for i in range(lineLength):
        tx = data[i * length:(i + 1) * length]
        if tx != u'':
            splitData.append(tx)

    return splitData


def rand_draw(angleTuple=(-10, 10), texts=None, length=4, back=True):
    """
    
    @@texts: 文本集，如果texts为none,那么随机读取语料库的文本
    
    """
    SizeList = [20000]
    Size = random.choice(SizeList)
    Size = Size, Size
    if texts is None:
        # texts = read_text()
        texts = read_text()
    im = None
    if back:
        path = np.random.choice(backPaths)
        if random.randint(0, 100) < 50:
            im = None
        else:
            im = Image.open(path).resize(Size)

    im, boxes, chars, lineBoxes, lineChars = draw_box(texts, size=Size, im=im)
    center = im.size[0] / 2, im.size[1] / 2
    lineBoxes, angle = rotate_box_tramform(lineBoxes, center, angleTuple)  ##随机旋转一个角度
    return im.rotate(angle), lineBoxes, lineChars, angle


def merge_line_box(lineBoxes, textes):
    """
    按行合并box
    """
    boxes = []
    linetexts = []
    for i, lineBox in enumerate(lineBoxes):
        lineBox = np.array(lineBox)
        if len(lineBox) != 0:
            x0, y0 = lineBox[:, ::2].min(), lineBox[:, 1::2].min()

            x2, y2 = lineBox[:, ::2].max(), lineBox[:, 1::2].max()

            boxes.append([int(x0), int(y0), int(x2), int(y2)])
            linetexts.append(u''.join(textes[i]))
    return boxes, linetexts


def crop_img(im, boxes, textes, root):
    """
    按行将文本及数据存为本地
    @@im
    @@boxes:box
    @@textes
    @@root:存入的路径
    """
    #print('crop_img label count: ', len(textes))
    for i, box in enumerate(boxes):
        cropIm = im.crop(box)
        text = textes[i]
        write_img_text(cropIm, text, root)


def write_img_text(im, text, root='data/0'):
    """
    写入行文本
    """
    if not os.path.exists(root):
        os.makedirs(root)
    path_raw = os.path.join(root, uuid1().__str__())
    path_erase = os.path.join(root, uuid1().__str__())
    raw_imgPath = path_raw + '.png'
    raw_txtPath = path_raw + '.txt'
    erase_imgPath = path_raw + '.png'
    erase_txtPath = path_raw + '.txt'
    if len(text) <= maxLen and len(text) >= minLen or maxLen is None:
        global gt_count
        im_erase = erase(im)
        im_erase = erase(im_erase)
        im.save(raw_imgPath)
        gt_count += 1
        im_erase.save(erase_imgPath)
        gt_count += 1
        with open(raw_txtPath, 'w') as f:
            f.write(text.encode('utf-8'))
        with open(erase_txtPath, 'w') as f:
            f.write(text.encode('utf-8'))


import traceback


def get_img_text(angle=(-5, 5), root='data/0', length=10, back=True):
    try:
        im, boxes, textes, _ = rand_draw(angle, length=length, back=back)
        boxes, textes = merge_line_box(boxes, textes)
        crop_img(im, boxes, textes, root)
    except:
        # traceback.print_exc()
        pass


def get_img_char(angle=(-5, 5), root='data/0', length=10):
    try:
        im, boxes, textes, _ = rand_draw(angle, length=length)
        # boxes,textes = merge_line_box(boxes,textes)
        crop_img(im, sum(boxes, []), sum(textes, []), root)
    except:
        pass


backPaths = glob.glob('./bg_img/*.png')  ##背景图像
fonts = glob.glob('./fonts/*.*')  ##字体集
corpusPaths = glob.glob('./corpus/contract/*.txt')  ##语料库
maxLen = 6  ##每行字符个数
minLen = 4
gt_count = 0
from multiprocessing import Pool
from tqdm import tqdm
if __name__ == '__main__':
    maxLen = 15  # 最长字符长度
    minLen = 3  # 最短字符长度
    prefix = 'contract-underline-erase'
    def get_img_text_for_multiple_process2(i):
        get_img_text(angle=(-2, 2),
                     root='/media/task0x04/data/imageLine/data/chn/chn_{}_{}-{}_with_back'.format(prefix, minLen,
                                                                                                  maxLen),
                     back=True)
        return 0
    p = Pool()
    r = list(tqdm(p.imap(get_img_text_for_multiple_process2, range(10000)), total=10000))
    # maxLen =   ##每行字符个数
    # for i in range(8000):
    #     print(i)
    #     get_img_text(angle=(-1,1),root='/media/task0x04/data/imageLine/data/chn/3500-chn-8_noback', back=False)
    # maxLen = 10  ##每行字符个数
    # for i in range(1000):
    #     print(i)
    #     get_img_text(angle=(-1, 1), root='../imageLine/data/chn-6')
    # maxLen = 8  ##每行字符个数
    # for i in range(1000):
    #     print(i)
    #     get_img_text(angle=(-1, 1), root='../imageLine/data/chn-8')
    # maxLen = 10  ##每行字符个数
    # for i in range(1000):
    #     print(i)
    #     get_img_text(angle=(-1, 1), root='../imageLine/data/chn-10')
    print("done.")
