import os
import time
import cv2
import numpy as np
import pickle
import gzip
import glob
import logging
from skimage import img_as_bool

from .utils.graph_utils import (
    my_print, 
    my_skeletonize, 
    buildTree, 
    mergeEdges, 
    getEndNodes, 
    draw_simpleGraphs
)
from .utils.vectorization_utils import (
    line_extraction, 
    gene_list_index_map, 
    merge_list_1st
)
from .utils.image_utils import get_rect_p0p1

class CNNVE:
    """
    A class for image vectorization using the CNNVE method.

    Attributes:
        model: The loaded line extraction model.
    """
    def __init__(self, device="cuda"):
        """
        Initializes the CNNVE vectorizer.

        Args:
            device (str, optional): The device that is used.
        """
        print("Loading deep learning model (this may take 5-10 seconds on first run)...")
        self.model = line_extraction()
        print("✓ Model loaded successfully!\n")
        self.device = device

    def run(self, input_path: str):
        """
        Vectorize the input raster image or images in a folder.

        Args:
            input_path (str): a path to an input raster image or a folder with multiple images.
        """
        if os.path.isdir(input_path):
            self.bat_process(input_path)
        else:
            # For single file, we might need to adjust output path logic
            # But for now, let's assume bat_process logic handles directories
            # If input is a file, we can process it directly.
            outp = input_path.replace('.png', '_output') # Simple output dir generation
            if not os.path.exists(outp):
                os.makedirs(outp)
            self.process_single_image(input_path, outp)

    def bat_process(self, indir):
        """
        Batch processes all PNG files in a directory.

        For each file `*.png` in `indir`, it creates an output directory
        `indir/output/...` and calls `process_single_image`.

        Args:
            indir (str): The directory containing PNG files to process.
        """
        fs = glob.glob(indir + r'/*.png')
        for f in fs:
            outp = f.replace(indir, indir + 'output').replace('.png', '')
            os.makedirs(outp, exist_ok=True)

            try:
                self.process_single_image(f, outp, iter_id=0, merge_or_onepath={})
            except Exception as e:
                logging.exception('err', f, e)

    def process_single_image(self, fname, outdir, iter_id=0, merge_or_onepath={}): 
        """
        Main processing pipeline to build a graph, segment strokes, and merge them.

        Args:
            fname (str): The file path of the input PNG image.
            outdir (str): The directory to save output and intermediate files.
            iter_id (int, optional): Iteration ID. Defaults to 0.
            merge_or_onepath (dict, optional): Dictionary to pass merge/path info.
        """
        b_donot_iter=True#sig25==True
        iter_limit=1
        if iter_id> iter_limit:
            return
        if not (os.path.exists(fname)):
            my_print(fname,'not found')
            return 
        merge_th=0.2#0.2#0.5 
        b_save_sk=True#False#True

        my_print('merge_th=',merge_th,fname)
        dic={}    
        path_graph=fname.replace('.png','.graph') 
        path_sk=fname.replace('.png','.sk.jpg')
        path_graph2=fname.replace('.png','.graph2')
        path_graph_merge=fname.replace('.png','.graph_merge')
        
        if (os.path.exists(path_graph_merge)):
            my_print(fname,'Done')
            if iter_id==0:
                print(fname,outdir)
            
            # If we want to visualize existing results, we could load them here
            # But for now, just return if done
            return

        t_ske,t_bdtree,t_smptree,t_alg1_op,t_alg1,t_cnn=-1,-1,-1,-1,-1,-1
        
        if not (os.path.exists(path_graph)):#False:
            oimg = cv2.imread(fname,0)
            gray_img=oimg
            ori_img=oimg.copy()

            #gray_img[gray_img<250]=0
            gray_img[gray_img<128]=0
            gray_img[gray_img>=128]=255

            gray_img=255-gray_img

            img = img_as_bool(gray_img)

            print("[1/5] Starting skeletonization...")
            t0=time.time()
            timgSk=my_skeletonize(gray_img)
            t_ske=time.time()-t0
            print(f"      ✓ Skeletonization completed in {t_ske:.2f}s")
            imgSk = img_as_bool(timgSk)

            # Find a start pixel (not necessary)
            y=int(imgSk.shape[1]/2)
            x=np.where(imgSk[:,y]==True)
            if len(x[0])>0:
                x=x[0][0]
            else:
                xy=np.where(imgSk[:,:]==True)
                if len(xy[0]) > 0:
                    x,y=xy[0][1],xy[0][0]
                else:
                    # Handle empty skeleton
                    x, y = 0, 0 

            imgSkT = imgSk.copy()
            print("[2/5] Building pixel-level graph...")
            t0=time.time()
            graphs , imgB = buildTree(imgSkT,np.array([x,y]))
            t_bdtree=time.time()-t0
            print(f"      ✓ Built {len(graphs)} graph(s) in {t_bdtree:.2f}s")
            my_print("built %i graphs" % len(graphs))
            
            dic['graphs']=graphs
            dic['skeletonize']=timgSk 
            dic['input']=gray_img
            dic['ori_input']=ori_img
            dic['merge_or_onepath']=merge_or_onepath

            tdic=dic
        else:
            # Logic to load existing graph if needed, but for now we assume we rebuild
            # or raise if path_graph exists but we are here (logic from original code)
            raise Exception("Graph file exists but logic reached rebuild block")

        simpleGraphs = []
        print("[3/5] Simplifying graph structure...")
        t0=time.time()
        for g in graphs:
            newG = mergeEdges(g)
            simpleGraphs.append(
                { 
                    "graph": newG,
                    "endNodes": getEndNodes(newG)
                }
            )
        t_smptree=time.time()-t0
        print(f"      ✓ Graph simplification completed in {t_smptree:.2f}s")
        if iter_id==0:
            draw_simpleGraphs(simpleGraphs,timgSk,path_sk[:-4]+'_graph.jpg')

        if not (os.path.exists(path_graph2)):
            print("[4/5] Extracting stroke masks (this may take 30-60 seconds)...")
            t0=time.time()
            # Pass self.model to gene_list_index_map
            maskdic=gene_list_index_map(self.model, simpleGraphs,gray_img,outdir,th=merge_th,img_bool_sk=imgSk,iter_id=iter_id)
            t_cnn=time.time()-t0
            print(f"      ✓ Stroke extraction completed in {t_cnn:.2f}s")
            tdic['simpleGraphs']=simpleGraphs
            maskdic['simpleGraphs']=simpleGraphs
            tdic['raw_maskdic']=maskdic

        else:
            raise

        if not (os.path.exists(path_graph_merge)):
            print("[5/5] Merging stroke masks...")
            t0=time.time()
            maskdic=tdic['raw_maskdic']
            merge_list_1st(maskdic,merge_th,img_bool_sk=imgSk,iter_id=iter_id)
            merge_dic=maskdic
            t_alg1=time.time()-t0
            print(f"      ✓ Stroke merging completed in {t_alg1:.2f}s")


            if not (os.path.exists(path_graph_merge)):
                with gzip.open(path_graph_merge,'wb') as f:
                    pickle.dump(merge_dic, f)

            t0=time.time()
            t_alg1_op=time.time()-t0

            simpleGraphs,edge_dic,mask_dic,G_merge_list=merge_dic['simpleGraphs'],merge_dic['EdgeDic'],merge_dic['SplitMaskDic'],merge_dic['MergeList']
            print(fname,',{0},{1},{2},{3},{4},{5}'.format(t_ske,t_bdtree,t_smptree,t_cnn,t_alg1,(t_ske+t_bdtree+t_smptree+t_alg1+t_cnn)))
            print(f"\n✓ Vectorization complete! Graph saved to: {path_graph_merge}")
            return
