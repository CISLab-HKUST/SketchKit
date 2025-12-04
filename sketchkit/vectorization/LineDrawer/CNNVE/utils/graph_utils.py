import numpy as np
import networkx as nx
import collections
import cv2
import skimage.morphology as morphology
from skimage import img_as_bool
import os
import pickle

def my_print(*args, **kwargs):
    """
    Custom print function that can be toggled by a boolean flag.
    
    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """
    b_time_only=False  # Set to False to show progress information
    if not b_time_only:
        print(*args, **kwargs)

def my_skeletonize(gray_img_blackbg,m_th=128):
    """
    Performs skeletonization on a grayscale image with a black background.

    The image is first binarized using the threshold `m_th`. It then applies
    OpenCV's thinning algorithm followed by skimage's medial axis transform.

    Args:
        gray_img_blackbg (np.ndarray): The input grayscale image. Assumes
            the object is light and the background is dark.
        m_th (int, optional): The threshold value for binarization.
            Defaults to 128.

    Returns:
        np.ndarray: A boolean array representing the skeleton (medial axis).
    """
    gray_img_blackbg[gray_img_blackbg<=m_th]=0
    gray_img_blackbg[gray_img_blackbg>m_th]=255
    gray_img=gray_img_blackbg
    type=1

    try:
        timgSk = cv2.ximgproc.thinning(gray_img)
    except AttributeError:
        # Fallback to skimage if opencv-contrib is not installed
        # thinning expects boolean or 0/1, gray_img is 0/255
        timgSk = morphology.thin(gray_img > 128)
        timgSk = (timgSk * 255).astype(np.uint8)

    img_m2 = img_as_bool(timgSk)
    npu8_sk=morphology.medial_axis(img_m2)
    return npu8_sk

class Vertex:
    """
    Represents a vertex (node) in a graph, typically a pixel in a skeleton.

    Attributes:
        point (np.ndarray): The (y, x) coordinates of the vertex.
        degree (int): The degree of the vertex (number of connected edges).
        edges (list): A list of Edge objects connected to this vertex.
        visited (bool): A flag used in graph traversal algorithms.
    """
    def __init__(self,point, degree=0, edges=None):
        """
        Initializes a new Vertex.

        Args:
            point (list or np.ndarray): The (y, x) coordinates.
            degree (int, optional): The degree of the vertex. Defaults to 0.
            edges (list, optional): A list of pre-defined edges. Defaults to None.
        """
        self.point= np.asarray(point)
        self.degree=degree
        self.edges=[]
        self.visited=False
        if edges is not None:
            self.edges=edges
    
    def __str__(self):
        """Returns the string representation of the vertex (its coordinates)."""
        return str(self.point)
    
class Edge:
    """
    Represents an edge in a graph, connecting two Vertex objects.

    Attributes:
        start (Vertex): The starting vertex of the edge.
        end (Vertex): The ending vertex of the edge.
        pixels (list): A list of (y, x) coordinate points that make up
            the path of this edge.
        visited (bool): A flag used in graph traversal algorithms.
    """
    
    def __init__(self,start,end=None,pixels=None):
        """
        Initializes a new Edge.

        Args:
            start (Vertex): The starting vertex.
            end (Vertex, optional): The ending vertex. Defaults to None.
            pixels (list, optional): A list of (y, x) pixel coordinates
                forming the edge. Defaults to None.
        """
        self.start=start
        self.end=end
        self.pixels = []
        if pixels is not None:
            self.pixels=pixels
        self.visited=False

def merge_list(L):
    """
    (Deprecated/Legacy) Merges sets in a list that share common elements.

    Args:
        L (list[set]): A list of sets.

    Returns:
        list[set]: A list of merged sets.
    """
    length = len(L)
    for i in range(1, length):
        for j in range(i):
            if L[i] == {0} or L[j] == {0}:
                continue
            x = L[i].union(L[j])
            y = len(L[i]) + len(L[j])
            if len(x) < y:
                L[i] = x
                L[j] = {0}

    return [i for i in L if i != {0}]

def merge_list_G(lst):
    """
    Merges lists of integers that share common elements (transitive closure).

    Uses a NetworkX graph to find connected components. For a list of
    pairs `[[1, 2], [3, 4], [2, 3]]`, this will find that 1, 2, 3, and 4
    are all connected and return `[[1, 2, 3, 4]]`.

    Args:
        lst (list[list[int]]): A list of lists, where each inner list
            represents a group or a pair of connected items.

    Returns:
        list[list[int]]: A list of the final merged component groups.
    """
    l=lst
    G = nx.Graph()
    G.add_nodes_from(sum(l, []))
    q = [[(s[i],s[i+1]) for i in range(len(s)-1)] for s in l]
    for i in q:
        G.add_edges_from(i)
    return [list(i) for i in nx.connected_components(G)]

def buildTree(img,start=None):
    """
    Builds a graph representation from a skeleton image.

    Traverses the skeleton image using BFS to construct a graph where
    pixels are vertices and 8-way connections are edges. If the image
    contains disconnected components, it will generate a list of graphs.

    Args:
        img (np.ndarray): A binary/boolean skeleton image (True or 255
            is foreground). The image will be modified in place.
        start (np.ndarray, optional): The (y, x) starting coordinate for
            the traversal. If None, it finds the first available
            foreground pixel.

    Returns:
        tuple:
            - list[nx.Graph]: A list of generated NetworkX graphs.
            - np.ndarray: The modified image with visited pixels set to False.
    """
    
    # copy image since we set visited pixels to black
    img = img.copy()
    shape = img.shape
    nWhitePixels = np.sum(img)
    my_print('buildTree start=',start,nWhitePixels)
    # neighbor offsets (8 nbors)
    nbPxOff = np.array([ [-1,-1], [-1,0] , [-1,1] ,
                         [0,-1] ,           [0,1]  ,
                         [1,-1] , [1,0]  , [1,1]
                       ])
    
    queue = collections.deque()
    
    # a list of all graphs extracted from the skeleton
    graphs = []
    
    blackedPixels = 0
    # we build our graph as long as we have not blacked all white pixels!
    #while nWhitePixels != blackedPixels:
    while nWhitePixels > blackedPixels:
        # if start not given: determine the first white pixel
        if start is None:
            if True:
                coord = np.where(img == 255)
                #coord = np.where(img == 0)
                it = np.nditer(img, flags=['multi_index'])
                my_print('it-',it[0],blackedPixels)
                
                while not it[0]:
                #while not it.finished:
                    it.iternext()
                start = it.multi_index      
            else:
                coord = np.where(img == 255)
                #if not b_time_only:
                #    my_print('start',coord)

                #start = np.array([coord[0][1],coord[0][0]])
                #my_print('start',start)
        
        startV = Vertex(start)
        queue.append(startV)
        my_print("Start vertex: ", startV)
        
        # set start pixel to False (visited)
        img[startV.point[0],startV.point[1]] = False
        blackedPixels +=1
            
        # create a new graph
        G=nx.Graph()
        G.add_node(startV)
        
        # build graph in a breath-first manner by adding 
        # new nodes to the right and popping handled nodes to the left in queue
        while len(queue):
            currV = queue[0]; # get current vertex
            #my_print("Current vertex: ", currV)

            # check all neigboor pixels 
            for nbOff in nbPxOff:
                
                # pixel index
                pxIdx = currV.point + nbOff

                if (pxIdx[0] < 0 or pxIdx[0] >= shape[0]) or (pxIdx[1] < 0 or pxIdx[1] >= shape[1]):
                    continue; # current neigbor pixel out of image

                if img[pxIdx[0],pxIdx[1]] :
                    #my_print( "nb: ", pxIdx, " white ")
                    # pixel is white
                    newV = Vertex( [pxIdx[0],pxIdx[1]] )
                    
                    # add edge from currV <-> newV
                    G.add_edge(currV,newV, object=Edge(currV,newV))
                    #G.add_edge(newV,currV)

                    # add node newV
                    G.add_node(newV)

                    # push vertex to queue
                    queue.append(newV)

                    # set neighbor pixel to black
                    img[pxIdx[0],pxIdx[1]] = False
                    blackedPixels +=1

            # pop currV
            queue.popleft()
        #end while
        
        # empty queue 
        # current graph is finished ->store it
        graphs.append(G)
        
        # reset start
        start = None
        
    #end while
        
    return graphs, img

def getEndNodes_sk(g):
    """
    Get all nodes with a degree of 1 (end nodes) from a graph.

    Args:
        g (nx.Graph): The input graph.

    Returns:
        list[Vertex]: A list of end nodes.
    """
    return [ n for n in nx.nodes_iter(g) if nx.degree(g,n) == 1 ]
    
def getEndNodes(g):
    """
    Gets key nodes (endpoints and junctions) from a graph.

    Key nodes are defined as nodes with degree 1 (endpoints) or
    degree > 2 (junctions). Includes fallback logic if no such
    nodes are found.

    Args:
        g (nx.Graph): The input NetworkX graph.

    Returns:
        list[Vertex]: A list of key Vertex objects.
    """
    #return [ n for n in nx.nodes_iter(g) if nx.degree(g,n) == 1 ]
    
    g1=[ n for n in list(nx.nodes(g)) if nx.degree(g,n) == 1 ]
    g2=[ n for n in list(nx.nodes(g)) if nx.degree(g,n) >2  ]
    #my_print('getEndNodes',len(g1),len(g2),g1,g2)
    gg=g1+g2
    if len(gg)==0:
        lst=list(nx.nodes(g))
        n_g=len(list(nx.nodes(g)))
        my_print(n_g)
        if n_g>2:
            gg=lst[[0,n_g//2]]
        else:
            gg=[lst[0]]
    return gg

def mergeEdges(graph):
    """
    Simplifies a graph by merging edges connected by degree-2 nodes.

    This function iterates through the graph and "dissolves" nodes that
    have a degree of 2 (i.e., are part of a simple path). The two edges
    connected to such a node are merged into a single edge, and their
    pixel lists are concatenated.

    Args:
        graph (nx.Graph): The input graph, typically from `buildTree`.

    Returns:
        nx.Graph: A simplified graph where paths of degree-2 nodes have
            been contracted into single edges.
    """
    
    # copy the graph
    g = graph.copy()

    startNodes = [ startN for startN in g.nodes() if nx.degree(g,startN) != 2] 
        
    
    for v0 in startNodes:
        
        # start a line traversal from each neighbor
        startNNbs = list(nx.neighbors(g,v0))
        
        if not len(startNNbs):
            continue
        
        counter = 0
        v1 = startNNbs[counter] # next nb of v0
        while True :
            
            if nx.degree(g,v1) == 2:
                # we have a node which has 2 edges = this is a line segement
                # make new edge from the two neighbors
                nbs = list(nx.neighbors(g,v1))
                
                # if the first neihbor is not n, make it so!
                if nbs[0] != v0:
                    nbs.reverse()
                
                
                pxL0 = g[v0][v1]["object"].pixels    # the pixel list of the edge 0
                pxL1 = g[v1][nbs[1]]["object"].pixels # the pixel list of the edge 1

                # fuse the pixel list from right and left and add our pixel n.point
                g.add_edge(v0,nbs[1], 
                           object=Edge(v0,nbs[1],pixels = pxL0 + [ v1.point ] + pxL1 )
                           )

                # delete the node n
                g.remove_node(v1)
                
                # set v1 to new left node
                v1 = nbs[1]
                        
            else:
                counter +=1
                if counter == len(startNNbs):
                    break;
                v1 = startNNbs[counter] # next nb of v0
                                
    # weight the edges according to their number of pixels
    for u,v,o in g.edges(data="object"):
        g[u][v]["weight"] = len(o.pixels)
        if len(o.pixels)==0:
            #my_print('len(o.pixels)',u,v,o.pixels)
            g[u][v]["weight"] = 0.1
    
    return g

def simple_graph_gene(single_stroke_mask,outpath='',is_blackbg=True, get_graph=True):#get_segment_stroke_with_endpoints_rebuild #RET simpleGraphs,imgSk,(xmin,xmax,ymin,ymax)
    """
    Generates a simplified graph from a single stroke mask.

    This function crops the mask to its bounding box, skeletonizes it,
    builds a detailed pixel-graph, and then simplifies that graph by
    merging nodes of degree 2.

    Args:
        single_stroke_mask (np.ndarray): A grayscale image of a single stroke.
        outpath (str, optional): A path for (currently disabled) caching.
            Defaults to ''.
        is_blackbg (bool, optional): True if the mask has a black background.
            Defaults to True.
        get_graph (bool, optional): If False, stops after skeletonization
            and returns no graph. Defaults to True.

    Returns:
        tuple:
            - list[dict] or None: A list of simplified graph dictionaries.
              Each dict contains the "graph", "endNodes", and "crossNodes".
            - np.ndarray or None: The boolean skeleton image.
            - tuple or None: The bounding box (xmin, xmax, ymin, ymax).
    """
    from .image_utils import get_rect_p0p1
    
    if (os.path.exists(outpath) and False):
        with open(outpath,'rb') as ff:
            outb=ff.read()
            res_dic=pickle.loads(outb)
            #svg_sub=autosvg.get_vk_svgtxt(res_dic)
            return svg_sub
    b_vis=False#True#False
    
    
    if b_vis:
        res=cv2.merge([single_stroke_mask,single_stroke_mask,single_stroke_mask])
    #if is_background_black(single_stroke_mask):
    if is_blackbg:
        single_stroke_mask_rec=255-single_stroke_mask
    else:
        single_stroke_mask_rec=single_stroke_mask
    #print(single_stroke_mask.shape,'-0',np.unique(single_stroke_mask))      
    xmin,xmax,ymin,ymax=get_rect_p0p1(single_stroke_mask_rec)
    #print('edges_fits_ret_one-get_rect_p0p1',xmin,xmax,ymin,ymax)
    if xmin is None or ymax-ymin==0 or xmax-xmin==0:
        print('simple_graph_gene-err',xmin,xmax,ymin,ymax)
        return None,None,None
    gray_img=single_stroke_mask[ymin:ymax,xmin:xmax]

    if is_blackbg:
        imgSk = gray_img>0#black bg
    else:
        imgSk = (255-gray_img)>0

    if is_blackbg:
        imgSk = my_skeletonize(gray_img)#cv2.ximgproc.thinning(gray_img)
    else:     
        imgSk = my_skeletonize(255-gray_img)#cv2.ximgproc.thinning(255-gray_img)

    if not get_graph:
        return None,imgSk,(xmin,xmax,ymin,ymax)
    y=int(imgSk.shape[1]/2)
    x=np.where(imgSk[:,y]==True)
    
    if len(x[0])>0:
        x=x[0][0]
    else:
        xy=np.where(imgSk[:,:]==True)


        if len(xy)==0:
            return None,None,None
        if len(xy[0])==0:
            return None,None,None
        x,y=xy[0][0],xy[1][0]
    imgSkT = imgSk.copy()
    graphs,imgB = buildTree(imgSkT,np.array([x,y]))#get_segment_stroke_with_endpoints_rebuild
    
    simpleGraphs = []
    end_nodes_all=[]
    for g0 in graphs:
        newG = mergeEdges(g0)#mergeEdges(g["graph"])
        g=newG
        node_list=list(nx.nodes(g))
        end_nodes=[ n for n in node_list if nx.degree(g,n) == 1]
        cross_nodes=[ n for n in node_list if nx.degree(g,n) > 2]
        my_print('end_nodes',len(end_nodes))
        if len(end_nodes)==0:
            end_nodes=[ n for n in node_list if nx.degree(g,n) >= 1]
        if len(end_nodes)==0:
            my_print('node_list',len(node_list))
            end_nodes=[node_list[0]]
        if len(end_nodes)>0:
            end_nodes_all+=end_nodes
        simpleGraphs.append(
            { 
                "graph": newG,
                "endNodes": end_nodes,
                "crossNodes": cross_nodes
            }
        )
    return simpleGraphs,imgSk,(xmin,xmax,ymin,ymax)   #simple_graph_gene 

def draw_simpleGraphs(simpleGraphs,timgSk,outfname):
    """
    Visualizes the simplified graphs on top of their skeleton.

    Draws the skeleton, marks end nodes (degree 1) in red, and
    cross nodes (degree > 2) in blue.

    Args:
        simpleGraphs (list[dict]): A list of graph dictionaries, as
            generated by `simple_graph_gene`.
        timgSk (np.ndarray): The boolean skeleton image.
        outfname (str): The file path to save the visualization.
    """
    # Convert boolean skeleton to uint8 to avoid depth warning
    img=255-np.where(timgSk,255,0).astype(np.uint8)
    #print(np.unique(timgSk))
    imc3=cv2.merge([img,img,img])
    thickness=3
    lineType=8
    for index,gdic in enumerate(simpleGraphs):
        g = gdic["graph"]
        startNodes = [ startN for startN in g.nodes() if nx.degree(g,startN) == 1] 
        cNodes = [ startN for startN in g.nodes() if nx.degree(g,startN) > 2] 
        for v in cNodes:
            x,y=v.point[1],v.point[0]
            point_int=(x,y)
            tmp_color=(0,0,255)
            cv2.circle(imc3, point_int, thickness, tmp_color, -1, lineType)
        for v in startNodes:   
            x,y=v.point[1],v.point[0]
            point_int=(x,y)
            tmp_color=(255,0,0)
            cv2.circle(imc3, point_int, thickness, tmp_color, -1, lineType)
    cv2.imwrite(outfname,imc3)
