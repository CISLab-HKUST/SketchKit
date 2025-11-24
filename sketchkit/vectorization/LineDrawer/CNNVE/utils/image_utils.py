import numpy as np
import cv2
import skimage.io
import skimage
import imageio

def is_background_black_undopainst(image, threshold=0.6):
    """
    Checks if an image's background is predominantly black.

    Note: This function modifies the input `image` by binarizing it.
    It checks if the ratio of black pixels (<= 127) exceeds the threshold.

    Args:
        image (np.ndarray): The input image (will be modified).
        threshold (float, optional): The ratio of black pixels to
            consider the background black. Defaults to 0.6.

    Returns:
        bool: True if the background is predominantly black, False otherwise.
    """
    mth=127
    image[image>mth]=255
    image[image<=mth]=0
    mask=(image==0)

    ratio = np.sum(mask) / (image.shape[0] * image.shape[1])
    return ratio > threshold

def is_background_black(image, threshold=0.8):
    """
    Checks if an image's background is predominantly black.

    Checks if the ratio of pure black pixels (value 0) exceeds the threshold.

    Args:
        image (np.ndarray): The input image.
        threshold (float, optional): The ratio of black pixels. Defaults to 0.8.

    Returns:
        bool: True if the background is predominantly black, False otherwise.
    """
    # 转换到HSV
    #hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 创建掩模
    #mask = cv2.inRange(hsv_image, (h_low, s_low, v_low), (h_high, s_high, v_high))
    mask=(image==0)
    # 计算掩模下的像素比例
    ratio = np.sum(mask) / (image.shape[0] * image.shape[1])
    return ratio > threshold

def get_connectedComponents(im_c1):
    """
    Counts the number of significant connected components in a binary image.

    Components with 3 or fewer pixels are ignored.

    Args:
        im_c1 (np.ndarray): A single-channel, 8-bit binary image.

    Returns:
        int: The number of components with more than 3 pixels.
    """
    connectivity=8
    output = cv2.connectedComponentsWithStats(im_c1, connectivity, cv2.CV_8UC1)#CV_32S
    (numLabels, labels, stats, centroids) = output
    
    cnt=0#numLabels-1
    for i in range(1,numLabels):
        if cv2.countNonZero((labels==i).astype(np.uint8))>3:
            cnt+=1

    return cnt

def loadImage(f):
    """
    Loads an image from a file path as a float array.

    Args:
        f (str): The file path.

    Returns:
        np.ndarray: The image as a float array (values 0.0 to 1.0).
    """
    return skimage.img_as_float(skimage.io.imread(f))

def save_gif(gif_path,images,duration=0.05):
    """
    Saves a list of images as a GIF.

    Args:
        gif_path (str): The file path to save the GIF to.
        images (list[np.ndarray]): A list of image frames.
        duration (float, optional): The duration (in seconds) for each
            frame. Defaults to 0.05.
    """
    imageio.mimsave(gif_path, images, duration=duration) 

def rect_gene(point,rect_w):
    """
    Generates a square bounding box (rectangle) centered at a point.

    Args:
        point (tuple[int, int]): The center point (x, y).
        rect_w (int): The "radius" of the box; the box will have
            width and height of 2 * rect_w.

    Returns:
        tuple[int, int, int, int]: The rectangle (x, y, w, h), where (x, y)
            is the top-left corner.
    """
    #rect_w=256#+32#128#256
    b_crop=True#False
    point_int=(int(point[0]),int(point[1]))
    new_rect=point_int[0]-rect_w,point_int[1]-rect_w,rect_w*2,rect_w*2   
    return new_rect

def get_rect_p0p1(cvimg):
    """
    Finds the bounding box of non-white pixels in a white-background image.

    Args:
        cvimg (np.ndarray): A grayscale image, assumed to have a
            white (or >127) background.

    Returns:
        tuple (int, int, int, int): The bounding box (xmin, xmax, ymin, ymax).
            Returns (None, None, None, None) if no non-white pixels are found.
    """
    gray = cvimg
    #_, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    binary=gray
    black_points = np.where(binary < 127)  
    #print(black_points)
    if len(black_points[0])==0:
        return (None, None, None, None)
    xmin,xmax=np.min(black_points[1]),np.max(black_points[1])
    ymin,ymax=np.min(black_points[0]),np.max(black_points[0])
    #print(xmin,xmax,ymin,ymax)
    return xmin,xmax,ymin,ymax
    #return (xmin, ymin), (xmax, ymax)
    #w,h=xmax-xmin, ymax-ymin
    
    return (xmin, ymin, xmax, ymax)
    #return (xmin, ymin, xmax-xmin, ymax-ymin)

def get_rect_p0p1_blackbg(cvimg):
    """
    Finds the bounding box of non-black pixels in a black-background image.

    This is a wrapper around `get_rect_p0p1` that inverts the image first.

    Args:
        cvimg (np.ndarray): A grayscale image with a black background.

    Returns:
        tuple (int, int, int, int): The bounding box (xmin, xmax, ymin, ymax).
    """
    return get_rect_p0p1(255-cvimg)

def checkMergedTopo_Components(m1,m2,img_bool_sk,ii,jj,iter_id):
    """
    Checks if merging two masks preserves their topology.

    This heuristic checks if the non-overlapping parts of the masks
    (m1 - intersection, m2 - intersection) break into more components
    than they had originally. This is used to prevent invalid merges.

    Args:
        m1 (dict): Mask dictionary for the first stroke, must contain
            'update_mask'.
        m2 (dict): Mask dictionary for the second stroke, must contain
            'update_mask'.
        img_bool_sk (np.ndarray): The boolean skeleton of the entire image.
        ii (int): Index of the first mask (for debugging).
        jj (int): Index of the second mask (for debugging).
        iter_id (int): Iteration ID (for debugging).

    Returns:
        bool: True if the merge is topologically "safe", False otherwise.
    """
    b_vis=False#True
    b_print=False
    b_img_bool_sk=True
    m_th=50

    m1=m1['update_mask'][:,:,0].copy()
    m2=m2['update_mask'][:,:,0].copy()
    
    if b_img_bool_sk:#remove sk not in m1 or m2 (the rest is the mask for m1 or m2)
        sk_m1=img_bool_sk.copy()
        sk_m2=img_bool_sk.copy()
        mI12=img_bool_sk.copy()

        sk_m1[m1<m_th]=False
        sk_m2[m2<m_th]=False
    
    xmin, xmax,ymin, ymax=get_rect_p0p1(255-cv2.bitwise_or(m1,m2))
    #print(xmin, ymin, xmax, ymax)

    box_m1=m1[ymin:ymax,xmin:xmax]
    box_m2=m2[ymin:ymax,xmin:xmax]
    
    U12=cv2.bitwise_or(box_m1,box_m2)
    imskU12=np.where(mI12,255,0)[ymin:ymax,xmin:xmax]
    imskU12[U12<m_th]=0
    I12=cv2.bitwise_and(box_m1,box_m2)



    if b_img_bool_sk:
        box_mm1=sk_m1[ymin:ymax,xmin:xmax]
        box_mm2=sk_m2[ymin:ymax,xmin:xmax]
        xorU1=box_mm1.copy()
        xorU2=box_mm2.copy()
        xorU1[I12>m_th]=False#np.bitwise_xor(box_mm1,I12>m_th)
        xorU2[I12>m_th]=False#np.bitwise_xor(box_mm2,I12>m_th)
        
        box_mm1=np.where(box_mm1,255,0).astype(np.uint8)
        box_mm2=np.where(box_mm2,255,0).astype(np.uint8)
        xorU1=np.where(xorU1,255,0).astype(np.uint8)
        xorU2=np.where(xorU2,255,0).astype(np.uint8)
    else:
        xorU1=cv2.bitwise_xor(box_m1,I12)
        xorU2=cv2.bitwise_xor(box_m2,I12)
    
    
    if b_img_bool_sk:
        n1=get_connectedComponents(box_mm1)
        n2=get_connectedComponents(box_mm2)
    else:
        n1=get_connectedComponents(box_m1)
        n2=get_connectedComponents(box_m2)

    xn1=get_connectedComponents(xorU1)
    xn2=get_connectedComponents(xorU2)

    if xn1>n1 or xn2>n2:
        #print('checkMergedTopo_Components-NG')
        return False

    return True

def checkMergedTopo_no_increate_cross(m1,m2,img_bool_sk,ii,jj,iter_id):
    """
    Checks if merging masks would increase cross-nodes (currently stubbed).

    Note: This function is hardcoded to always return True. The intended
    logic (commented out in the source) appears to have been to generate
    graphs for m1, m2, and their union, then compare cross-node counts.

    Args:
        m1 (dict): (Unused) Mask dictionary for the first stroke.
        m2 (dict): (Unused) Mask dictionary for the second stroke.
        img_bool_sk (np.ndarray): (Unused) The boolean skeleton of the entire image.
        ii (int): (Unused) Index of the first mask.
        jj (int): (Unused) Index of the second mask.
        iter_id (int): (Unused) Iteration ID.

    Returns:
        bool: Always returns True.
    """
    return True
    # Implementation commented out in original code
    # from .graph_utils import simple_graph_gene
    # ...

def revise_mask_with_connectedComponents(im_c1,pixel_lst):#mask,graph[u][v]["object"].pixels get first region from prompts
    """
    Cleans a mask by keeping only the connected component touching the skeleton.

    Given a predicted mask (`im_c1`) and the list of skeleton pixels (`pixel_lst`)
    that generated it, this function finds all connected components in the mask.
    It then returns only the component that is "touched" by the first
    pixel in `pixel_lst`.

    Args:
        im_c1 (np.ndarray): The single-channel mask to revise (8-bit).
        pixel_lst (list[tuple]): The list of (y, x) skeleton pixels.

    Returns:
        np.ndarray: The revised 8-bit mask, containing only the selected
            connected component.
    """
    connectivity=8
    N=len(pixel_lst)
    #print(pixel_lst)
    output = cv2.connectedComponentsWithStats(im_c1, connectivity, cv2.CV_8U)#CV_32S
    (numLabels, labels, stats, centroids) = output
    if numLabels==2:
        img=np.where(labels==1,255,0)
        return np.array(img,np.uint8).reshape(im_c1.shape)
    #print('revise_mask_with_connectedComponents',numLabels-1)
    methodtype=1
    if methodtype==1:#get first region
        for i in range(N):
            x,y=pixel_lst[i][1],pixel_lst[i][0]
            if im_c1[y][x]==255:
                lb=labels[y][x]
                if lb>1:
                    pass
                    #my_print(lb,labels.shape)
                img=np.where(labels==lb,255,0)
                #h,w=im_c1.shape[:2]
                return np.array(img,np.uint8).reshape(im_c1.shape)#.reshape((h,w,1))
    elif methodtype==2:#get max region
        lst=[]
        for i in range(N):
            x,y=pixel_lst[i][1],pixel_lst[i][0]
            if im_c1[y][x]==255:
                lst.append(labels[y][x])
        if len(lst)==0:
            return np.array(im_c1,np.uint8)
        maxlb=max(lst,key=lst.count)
        print('Get max region',maxlb,len(lst),len(pixel_lst))
    #my_print('miss match connectedComponents(',im_c1.shape)
    return np.array(im_c1,np.uint8)

def criterion_mask_overlap(m_s,m_d,merge_th,i,j):#ret if_got_rate,rate
    """
    Calculates the overlap between two masks to check for merging.

    Calculates the intersection (I) and union (U) and pixel counts
    for each mask (c, d).
    The "rate" is max(I/c, I/d).
    A merge is considered valid only if the intersection `a` is above
    a minimum pixel threshold `I_th`.

    Args:
        m_s (np.ndarray): Source mask (binary).
        m_d (np.ndarray): Destination mask (binary).
        merge_th (float): (Unused) The overlap threshold.
        i (int): (Unused) Index of first mask.
        j (int): (Unused) Index of second mask.

    Returns:
        tuple (bool, float or None):
            - bool: True if the intersection is non-zero and above `I_th`.
            - float or None: The overlap rate, or None if the overlap
              is too small or zero.
    """
    th=merge_th#0.9#0.8#0.9#merge_th
    I_th=30#100#交点应该小于pi*r^2个点 线的半径记为r
    kernel = np.ones((3,3), np.uint8)
    b_vis=False#True False
    ii,jj=i,j


    I=cv2.bitwise_and(m_s,m_d)
    U=cv2.bitwise_or(m_s,m_d)
    a,b=cv2.countNonZero(I),cv2.countNonZero(U)
    c,d=cv2.countNonZero(m_s),cv2.countNonZero(m_d)
    if a*b*c*d==0:
        return False,None
    #my_print('IoU',i,j,a,b,c,d,a/b,a/c,a/d)
    rate=max(a/c, a/d)
    #print('criterion_mask_overlap-IoU',i,j,'rate=',rate,a,b,c,d,a/b,a/c,a/d)
    if I_th>a:
        return False,None

    return True,rate      

def color_gene(N=100):
    """
    Generates a list of N random colors.

    Args:
        N (int, optional): The number of colors to generate. Defaults to 100.

    Returns:
        list: A list of N random RGB tuples.
    """
    np.random.seed(1)
    colors = np.random.randint(0, 255, size=(N, 3))
    colors1 = [np.array(colors[i//2]) if i%2==0 else 255-np.array(colors[i//2]) for i in range(2*N) ]
    return colors1[:N]
