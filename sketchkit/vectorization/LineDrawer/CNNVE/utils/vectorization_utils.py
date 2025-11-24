import numpy as np
import cv2
import torch
import copy
import networkx as nx
from .graph_utils import my_print, merge_list_G, simple_graph_gene
from .image_utils import (
    rect_gene, 
    get_rect_p0p1_blackbg, 
    revise_mask_with_connectedComponents, 
    criterion_mask_overlap, 
    checkMergedTopo_Components, 
    checkMergedTopo_no_increate_cross
)
from ..network import model as pm

def line_extraction():
    """
    Initializes and loads the line extraction deep learning model.

    Loads the `stNet_double` model from `CNNVE.pm` and loads pre-trained
    weights ("500.pth") onto the available device (CUDA or CPU).

    Returns:
        pm.stNet_double: The initialized and loaded PyTorch model.
    """
    MODEL_NAME = '500'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = pm.stNet_double(DEVICE).to(DEVICE)
    model.load(MODEL_NAME)
    my_print('line_extraction load',MODEL_NAME)
    return model

def line_extraction_batch_with_mask(model,img_c1,point_list_xy,mask_list,rect_w=256):#bat 多edge
    """
    Performs batch line extraction using a model on specified masks.

    Takes a list of masks and their corresponding center points, generates
    bounding boxes, and runs them through the model's
    `get_result_batch_s2` method in batches.

    Args:
        model (pm.stNet_double): The loaded line extraction model.
        img_c1 (np.ndarray): The original single-channel image.
        point_list_xy (list[list[tuple]]): A list where each item is a
            list of (x, y) points. The first point is used as the center.
        mask_list (list[np.ndarray]): A list of mask images corresponding
            to `point_list_xy`.
        rect_w (int, optional): The "radius" for the bounding box.
            Defaults to 256.

    Returns:
        tuple:
            - tuple(np.ndarray, np.ndarray, np.ndarray): Model results
              (m1, m2, m0) concatenated from all batches.
            - list[tuple]: The list of generated rectangles (rect_lst).
    """
    #mask_img=np.zeros((img_c1.shape), np.uint8)
    #to do rect_crop and paste back
    tmp_color=(255,255,255)
    lineType=4
    thickness=3
    
    #rect_w=256#128#256
    #rect=(point_int[0]-rect_w,point_int[1]-rect_w,rect_w*2,rect_w*2)
    mask_lst=mask_list
    rect_lst=[]
    for xy_lst in point_list_xy:
        assert len(xy_lst)>0
        xy=xy_lst[0]
        point_int=(xy[0],xy[1])
        rect_lst.append(rect_gene(point_int,rect_w))
        
        

    assert len(mask_lst)==len(rect_lst)

    bn=640
    n_mask=len(mask_lst)
    if n_mask%bn==0 and n_mask>0:
        nt=(n_mask//bn)
    else:   
        nt=(n_mask//bn)+1
    r0_list,r1_list,m_list=[],[],[]
    for i in range(nt):
        mask_lst_sub= mask_lst[i*bn:min((i+1)*bn,n_mask)]
        rect_lst_sub= rect_lst[i*bn:min((i+1)*bn,n_mask)]
        s1,s2,s0=model.get_result_batch_s2(img_c1,mask_lst_sub,rect_lst_sub)
        r0_list.append(s1)
        r1_list.append(s2)
        m_list.append(s0)
    m1=np.concatenate( r0_list, axis=0)
    m2=np.concatenate( r1_list, axis=0)
    m0=np.concatenate( m_list, axis=0)
    return (m1,m2,m0),rect_lst

def stroke_extracion_multi(model, points, im_ori, b_with_remove=False):
    """
    Extracts strokes from an image using a model, given multiple center points.

    For each point in `points`, it creates a small circular mask,
    then runs the batch prediction `model.get_result_batch`.
    Finally, it combines all the resulting masks using bitwise OR.

    Args:
        model (pm.stNet_double): The loaded line extraction model.
        points (list[tuple]): A list of (x, y) center points.
        im_ori (np.ndarray): The original single-channel image.
        b_with_remove (bool, optional): (Unused) Defaults to False.

    Returns:
        np.ndarray: A (H, W, 1) mask combining all extracted strokes.
    """
    h,w=im_ori.shape[:2]
    n_pts=len(points)
    rect_w=256+32#128#256
    horw=min(h,w)//2
    if horw<rect_w:
        rect_w=(horw//4)*4
        
    b_crop=True#False
    new_rect=(0,0,1,1)
    mask_img=np.zeros((h,w), np.uint8)
    #mask_img_result=mask_img.copy()
    out_mask_img=mask_img.copy()
    tmp_color=(255,255,255)
    lineType=4
    thickness=3
    rect_lst=[]
    mask_lst=[]
    sk_img_c1=im_ori#self.get_masked_base_sketch()#self.parent_frm.input_sketch.copy()
    for point in points:
        point_int=(int(point[0]),int(point[1]))
        rect=(point_int[0]-rect_w,point_int[1]-rect_w,rect_w*2,rect_w*2)
        rect_lst.append(rect)
        a_mask_img=cv2.circle(mask_img.copy(), point_int, thickness, tmp_color, -1, lineType)
        mask_lst.append(a_mask_img)
        
    m1,m2,m0=model.get_result_batch(sk_img_c1,mask_lst,rect_lst)
    #bm1=mask_paste_back(m1,rect_lst,points,sk_img_c1)
    #bm2=mask_paste_back(m2,rect_lst,points,sk_img_c1)
    bm2=m2
    
    for m_t in bm2:
        out_mask_img=cv2.bitwise_or(out_mask_img,m_t)#, mask=m0)
    #im_out_full=255-cv2.merge([sk_img_c1,sk_img_c1,sk_img_c1])
    #im_out_full[out_mask_img>MASK_TH]=(0,255,0)
    
    return np.array(out_mask_img,np.uint8).reshape((h,w,1))

def stroke_extracion_multi_segments(model, points, im_ori, b_with_remove=False):
    """
    Extracts strokes, optimized for long segments of points.

    If the number of points is small, it falls back to
    `stroke_extracion_multi`. If large, it breaks the point list into
    overlapping segments, creates a polyline mask for each segment,
    and runs the batch prediction `model.get_result_batch_s2`.

    Args:
        model (pm.stNet_double): The loaded line extraction model.
        points (list[tuple]): A list of (x, y) points defining the stroke.
        im_ori (np.ndarray): The original single-channel image.
        b_with_remove (bool, optional): (Unused) Defaults to False.

    Returns:
        np.ndarray: A (H, W, 1) mask combining all extracted strokes.
    """
    h,w=im_ori.shape[:2]
    n_pts=len(points)
    if n_pts<8:#6
        #raise
        return stroke_extracion_multi(model, points, im_ori, b_with_remove).reshape((h,w,1))
    rect_w=256+32#128#256 #576*576
    horw=min(h,w)//2
    if horw<rect_w:
        rect_w=(horw//4)*4
    b_crop=True#False
    new_rect=(0,0,1,1)
    mask_img=np.zeros((h,w), np.uint8)
    #mask_img_result=mask_img.copy()
    out_mask_img=mask_img.copy()
    tmp_color=(255,255,255)
    lineType=4
    thickness=3
    rect_lst=[]
    mask_lst=[]
    sk_img_c1=im_ori
    
    
    if n_pts>20:
        s_size=n_pts//4
        merge_win=3
        mth=(80,20)
        #mth=(70,15)
        if n_pts >mth[0]:#80,
            seg_size=n_pts//mth[1]
            s_size=n_pts//seg_size
            #merge_win=3
        
        points_lst=[points[i:i + s_size] for i in range(0, n_pts, s_size)]
        points_lst2=[]
        for i in range(len(points_lst)-merge_win+1):
            tmp_lst=[]
            for j in range(merge_win):
                tmp_lst+=points_lst[i+j]
            points_lst2.append(tmp_lst)
    else:
        points_lst2=[points]
    for pts in points_lst2:
        #N_pt2=len(pts)
        #if N_pt2>#remove node neibor
        point=pts[len(pts)//2]#pts[0]#--v1
        point_int=(int(point[0]),int(point[1]))
        rect=(point_int[0]-rect_w,point_int[1]-rect_w,rect_w*2,rect_w*2)
        rect_lst.append(rect)
        blist =np.array(pts,np.int32).reshape(-1,1,2) 
        #my_print(blist)
        a_mask_img=cv2.polylines(mask_img.copy(), [blist], False, tmp_color, thickness=thickness)
        mask_lst.append(a_mask_img)
    m1,m2,m0=model.get_result_batch_s2(sk_img_c1,mask_lst,rect_lst)
    my_print('m1,m2,m0',m1.shape,m2.shape,m0.shape)
    #bm1=mask_paste_back(m1,rect_lst,points,sk_img_c1)
    #bm2=mask_paste_back(m2,rect_lst,points,sk_img_c1)
    bm2=m2
    
    for m_t in bm2:
        out_mask_img=cv2.bitwise_or(out_mask_img,m_t)#, mask=m0)
    #im_out_full=255-cv2.merge([sk_img_c1,sk_img_c1,sk_img_c1])
    #im_out_full[out_mask_img>MASK_TH]=(0,255,0)
    #self.stroke_hint_img=im_out_full
    
    return np.array(out_mask_img,np.uint8).reshape((h,w,1))

def dist_L2(points):
    """
    Calculates the total L2 (Euclidean) distance of a path.

    Args:
        points (list or np.ndarray): A list of (x, y) points.

    Returns:
        float: The total path length.
    """
    points=np.array(points)
    #print(len(points))
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)  
    return np.sum(distances)

MASK_TH=50

def truncate(points):
    """
    (Deprecated/Stub) Truncates the start and end of a point list.

    Note: This function is hardcoded to return the original list.
    The commented-out logic suggests an intent to remove a small
    number of points from the beginning and end of the list.

    Args:
        points (list): A list of points.

    Returns:
        list: The original, unmodified list of points.
    """
    return points

def update_mask_edge(model,ori_img,pixel_lst):#better?
    """
    Generates a stroke mask from a list of skeleton pixels.

    This function converts skeleton (y, x) pixels to (x, y) points,
    calculates the path length, and decides on an extraction strategy.
    -   If the path is very short, it uses the midpoint.
    -   Otherwise, it uses the (potentially truncated) full list of points.
    It calls `stroke_extracion_multi_segments` to get the final mask.

    Args:
        model (pm.stNet_double): The loaded line extraction model.
        ori_img (np.ndarray): The original single-channel image.
        pixel_lst (list[tuple]): A list of (y, x) skeleton pixels.

    Returns:
        np.ndarray: The (H, W, 1) generated stroke mask.
    """
    N=len(pixel_lst)
    points=[(a[1],a[0])for a in pixel_lst]
    dist=dist_L2(points)
    #print('update_mask_edge',N,dist)
    if N<3:
        h,w=ori_img.shape
        new_mask= np.zeros((h,w,1))
        #print(update_mask_edge,new_mask.shape)
        assert len(new_mask.shape)==3
        return new_mask 
    #elif N<40 and dist<30:
    elif N<8 and dist<30:
        new_mask= stroke_extracion_multi_segments(model,[points[N//2]],ori_img)#update_mask_edge
        assert len(new_mask.shape)==3
        return new_mask 
    #new_mask= stroke_extracion_multi_segments(model,points,ori_img)
    new_mask= stroke_extracion_multi_segments(model,truncate(points),ori_img)
    assert len(new_mask.shape)==3
    return new_mask

def update_mask(model,ori_img,mask_curr,pixel_lst):
    """
    (Deprecated/Heuristic) Updates a mask if it seems to mismatch the skeleton.

    Checks if a large number of skeleton pixels (`pixel_lst`) fall
    outside the current mask (`mask_curr`). If the mismatch is large,
    it re-generates the mask using `stroke_extracion_multi_segments`.

    Args:
        model (pm.stNet_double): The loaded line extraction model.
        ori_img (np.ndarray): The original single-channel image.
        mask_curr (np.ndarray): The existing (H, W, 1) mask.
        pixel_lst (list[tuple]): A list of (y, x) skeleton pixels.

    Returns:
        np.ndarray: The original mask or a newly generated one.
    """
    N=len(pixel_lst)
    points=[(a[1],a[0])for a in pixel_lst]
    #dist=dist_L2(points)
    if N<6:
        new_mask= stroke_extracion_multi_segments(model,[points[0]],ori_img)
        return new_mask 
    pmask=np.zeros((ori_img.shape), np.uint8)
    for i in range(N):
        x,y=pixel_lst[i][1],pixel_lst[i][0]
        pmask[y,x]=255
    I=cv2.bitwise_and(mask_curr,pmask)
    a=cv2.countNonZero(I)
    if N-a>20:
        #my_print('update_mask',a,N)
        
        new_mask= stroke_extracion_multi_segments(model,points,ori_img)
        #my_print('new_mask',new_mask.shape)
        return new_mask
    else:
        return mask_curr

def gene_list_index_map(model, simpleGraphs,im_ori_c1,outdir,th=0.2,img_bool_sk=None,iter_id=0):#merge_edge_loop np.sum修正为cv2, 找点而不算重合度,  gene_list_index_map -> update_mask_edge
    """
    Generates stroke masks for each edge in the simplified graphs.

    Iterates through all edges in `simpleGraphs`. For each edge:
    1.  Gets the skeleton pixel list.
    2.  If it's a simple component, gets the mask from `connectedComponents`.
    3.  Otherwise, generates a mask using the ML model (`update_mask_edge`).
    4.  Cleans the generated mask (`revise_mask_with_connectedComponents`).
    5.  Stores the mask and related info in a dictionary `mdic` indexed
        by a counter.
    6.  Creates a lookup table `edic` mapping edge names to the counter index.

    Args:
        model (pm.stNet_double): The loaded line extraction model.
        simpleGraphs (list[dict]): List of simplified graph dictionaries.
        im_ori_c1 (np.ndarray): The original binarized image (black bg).
        outdir (str): (Unused) Output directory.
        th (float, optional): (Unused) Threshold. Defaults to 0.2.
        img_bool_sk (np.ndarray, optional): (Unused) Boolean skeleton image.
        iter_id (int, optional): (Unused) Iteration ID.

    Returns:
        dict: A dictionary containing:
            - 'EdgeDic' (edic): Maps edge names to mask IDs.
            - 'SplitMaskDic' (mdic): Maps mask IDs to mask data.
            - 'MergeList': A list of all mask IDs, unmerged.
    """
    edic={}
    mdic={}
    bdic={}
    h,w=im_ori_c1.shape[:2]
    output = cv2.connectedComponentsWithStats(im_ori_c1, 8, cv2.CV_8U)#CV_32S
    (numLabels, labels, stats, centroids) = output
    #print('numLabels',numLabels,len(simpleGraphs)) 
    #assert numLabels==len(simpleGraphs)+1
    cnt=-1
    m_th=50
    b_th=float(10)
    b_crit_curve=False#True
    #criterion_fun=criterion_mask_overlap 
    for i,g in enumerate(simpleGraphs):
        graph = g["graph"]
        b_single_Components=False
        #b_single_split=False
        #if len(graph.nodes())==2:
        endNodes = [ startN for startN in graph.nodes() if nx.degree(graph,startN) == 1] 
        if len(endNodes)<=3:
            if len(endNodes)==2:
                b_single_Components=True
        
        for u,v in graph.edges:
            name='%d,%d_%d,%d'%(u.point[1],u.point[0],v.point[1],v.point[0])
            name2='%d,%d_%d,%d'%(v.point[1],v.point[0],u.point[1],u.point[0])
            px0 = np.array( graph[u][v]["object"].pixels )
            #if len(px0)>0:
            if len(px0)>2:
                #m=graph[u][v]["big_mask2"]
                pixel_lst = px0#np.array( graph[u][v]["object"].pixels )
                #print(len(px0),pixel_lst)
                #m[m>=m_th]=255
                #m[m<m_th]=0
                if b_single_Components:
                    y,x=px0[0]
                    lb=labels[y,x]
                    #print(lb,labels.shape)
                    m2=(labels==lb).astype(np.uint8)*255
                    m2=m2.reshape((h,w,1))
                    #cv2.imwrite(outdir+'_%s.png'%name,m2)#
                    #print('b_single_Components',lb,m2.shape)
                else:
                    m=update_mask_edge(model,im_ori_c1,pixel_lst)
                    #m=update_mask(model,im_ori_c1,m,pixel_lst)
                    
                    m[m>=m_th]=255
                    m[m<m_th]=0
                    m2=revise_mask_with_connectedComponents(m,pixel_lst)
                
                mrec=get_rect_p0p1_blackbg(m2)
                #print(m2.shape,m.shape)
                #m2=m.copy()#noise
                assert len(m2.shape)==3
                cnt+=1
                #if len(pixel_lst)>100 and False:#896,597_854,196
                if  '896,597' in name and len(pixel_lst)>100 and False:#'896,597_854,196'
                    print('pixel_lst',len(pixel_lst))
                
                    ctmp=cv2.merge([m2,im_ori_c1,im_ori_c1])
                    for i in range(len(pixel_lst)):
                        x,y=pixel_lst[i][1],pixel_lst[i][0]
                        ctmp[y,x]=(0,0,255)
                    cv2.imwrite('xorU/'+name+'.png',ctmp)

                mdic[cnt]={'update_mask':m2,'point_list':pixel_lst,'boundary_box':mrec,'one_path_component':b_single_Components}#x,y=pixel_lst[i][1],pixel_lst[i][0]
                if b_crit_curve:   
                    px = np.array([ u.point ] + graph[u][v]["object"].pixels + [ v.point ]) 
                    ##print('graph',i,px.shape)#,len(beziers))
                    if len(px)>=2+2:
                        # beziers = fitCurve(px, b_th )
                        # bdic[cnt]={'edge_beziers':beziers,'pixels':px,'start_pt':u,'end_pt':v}
                        pass
                
            edic[name]=cnt
            edic[name2]=cnt
            #cnt+=1
    c_list=[]
    checked_list=[]
    
    id_dic={}
    
    
    single_list=[]
    for a in mdic.keys():   
        if a not in id_dic.keys():
            single_list.append([a])
        
    G_merge_list=merge_list_G(c_list)+single_list
    #print('single_list',single_list,G_merge_list)      
    #print('G_merge_list=',G_merge_list)
    dic_a={'EdgeDic':edic,'SplitMaskDic':mdic,'MergeList':G_merge_list}
    #vis_merge_result(im_ori_c1,G_merge_list,mdic)
    
    return dic_a

def merge_list_1st(mask_dic,th,img_bool_sk=None,iter_id=0):#len(c_list)>0 is iter
    """
    Performs the first pass of stroke merging.

    Iterates through all junctions (nodes with degree > 2) in the
    simplified graphs. For each pair of edges connected to that junction,
    it checks if their corresponding masks overlap significantly
    (`criterion_mask_overlap`) and pass a topological check
    (`checkMergedTopo_Components` and `checkMergedTopo_no_increate_cross`).
    
    If they pass, they are added to a component list (`c_list`) for merging.
    This function updates the `mask_dic` with the merge results.

    Args:
        mask_dic (dict): The main dictionary containing 'simpleGraphs',
            'SplitMaskDic', and 'EdgeDic'. This dict is modified in place.
        th (float): The overlap rate threshold for merging.
        img_bool_sk (np.ndarray, optional): The boolean skeleton image.
        iter_id (int, optional): The current iteration ID.

    Returns:
        list[list[int]]: The list of merged pairs (`c_list`).
    """
    #{'EdgeDic':edic,'SplitMaskDic':mdic,'MergeList':G_merge_list}
    simpleGraphs=mask_dic['simpleGraphs']
    
    mdic_ori=mask_dic['SplitMaskDic']
    edic=mask_dic['EdgeDic'] 
    id_dic={}
    checked_list=[]
    #c_list=[]
    rate_dic={}
    if 'MergedPairList' in mask_dic.keys():
        c_list=mask_dic['MergedPairList']
    else:
        c_list=[]
    if iter_id==0:
        c_list=[]
    if len(c_list)==0:
        mdic=mdic_ori
        print("new merge_list_1st",len(mdic.keys()))#simple_graph_gene
        #update single stroke graph
        if False:
            for kk in mdic.keys():
                m_simpleGraphs,imgSk,rect=simple_graph_gene(mdic[kk]['update_mask'][:,:,0],is_blackbg=True)
                mdic[kk]['m_simpleGraphs']=m_simpleGraphs
                mdic[kk]['mask_skeleton']=imgSk
                mdic[kk]['mask_rect']=rect
                pass
    else:
        mdic=copy.deepcopy(mdic_ori)
        G_merge_list=merge_list_G(c_list)
        #update mask merge 
        print("update mask merge",len(mdic.keys()),G_merge_list)
        for mlist in G_merge_list:
            mcnt=-1
            if len(mlist)>1:
                tmask=None
                orishape=None
                for ii in mlist:
                    mcnt+=1
                    if tmask is None:
                        tmask=mdic[ii]['update_mask'][:,:,0]
                        orishape=tmask.shape
                        #assert 3==len(tmask.shape)
                    else:
                        print('merge-test',len(tmask.shape),tmask.shape,mdic[ii]['update_mask'].shape)
                        tmask=cv2.bitwise_or(tmask,mdic[ii]['update_mask'][:,:,0])

                        #print(len(tmask.shape))
                        #assert len(tmask.shape)==3
                ht,wt=tmask.shape
                tmask=tmask.reshape((ht,wt,1))
                
                for ii in mlist:#merge_update
                    mdic[ii]['update_mask']=tmask
                    id_dic[ii]=ii
                    
    for index,gdic in enumerate(simpleGraphs):
        #print('gdic',gdic,len(simpleGraphs))#.keys()
        g = gdic["graph"]
        #startNodes = [ startN for startN in g.nodes() if nx.degree(g,startN) != 2] 
        startNodes = [ startN for startN in g.nodes() if nx.degree(g,startN) > 2] 
        for v0 in startNodes:
            startNNbs = list(nx.neighbors(g,v0))
            n_NNbs=len(startNNbs)
            #print(v0,'NNbs=',len(startNNbs))
            if not n_NNbs:
                continue
            if n_NNbs==1:
                continue
            for i in range(n_NNbs):
                for j in range(n_NNbs):
                    if i>=j:
                        continue
                    v1=startNNbs[i]
                    v2=startNNbs[j]
                    namev1='%d,%d_%d,%d'%(v0.point[1],v0.point[0],v1.point[1],v1.point[0])
                    namev2='%d,%d_%d,%d'%(v0.point[1],v0.point[0],v2.point[1],v2.point[0])
                    ii=edic[namev1]
                    jj=edic[namev2]
                    if ii not in mdic.keys() or jj not in mdic.keys():
                        continue
                    aa=mdic[ii]
                    bb=mdic[jj]
                    #if [i,j] in c_list or [j,i] in c_list:
                    if [ii,jj] in c_list or [jj,ii] in checked_list:
                        continue
                    checked_list.append([ii,jj])
                    
                    
                    res=criterion_mask_overlap(aa['update_mask'].copy(),bb['update_mask'].copy(),th,ii,jj)
                    if res[0]:#== res[0]>th
                        if res[1] is not None:
                            rate=res[1]
                            if rate >0.95:#or checkMergedTopo_Components(aa,bb,img_bool_sk,ii,jj)
                                c_list.append([ii,jj])
                                id_dic[ii]=ii
                                id_dic[jj]=jj
                            elif rate>th:
                                if checkMergedTopo_Components(aa,bb,img_bool_sk,ii,jj,iter_id):
                                    if checkMergedTopo_no_increate_cross(aa,bb,img_bool_sk,ii,jj,iter_id):
                                        c_list.append([ii,jj])
                                        id_dic[ii]=ii
                                        id_dic[jj]=jj
                    if res[1] is not None: 
                        rate_namev1='%d,%d'%(ii,jj)    
                        rate_dic[rate_namev1]=res[1]
    single_list=[]
    for a in mdic.keys():   
        if a not in id_dic.keys():
            single_list.append([a])
    G_merge_list=merge_list_G(c_list)+single_list
    #print('no sorted G_merge_list=' ,G_merge_list)
    G_merge_list = sorted(G_merge_list, key=lambda sub: min(sub))
    #print('sorted G_merge_list=' ,G_merge_list)

    mask_dic['MergeList']=G_merge_list
    mask_dic['MergedPairList']=c_list
    return c_list
