import numpy as np
import os,pickle
import matplotlib.pylab as plt
from tqdm import tqdm
import torch
import pathlib

def get_pos(path):
    """Given a .zarr file, get the x,y um positions of the stage"""
    xml_file = os.path.splitext(path)[0] + '.xml'
    txt = open(xml_file, 'r').read()
    tag = '<stage_position type="custom">'
    x, y = eval(txt.split(tag)[-1].split('</')[0])
    return x,y

def get_im_from_Xh(Xh,resc=5):
    """Given a numpy array Xh of fitted spots, return an 3D rescled image (by resc) with 1s where localizations are found"""
    X = np.round(Xh[:,:3]/resc).astype(int)
    sz = np.max(X,axis=0)
    imf = np.zeros(sz+1,dtype=np.float32)
    imf[tuple(X.T)]=1
    return imf
from scipy.spatial import KDTree
def get_Xtzxy(X,X_ref,tzxy0,resc,target=3):
    """Refine the subpixel drift between points <X> and <X_ref> starting with drift <tzxy0> and iterating to traget precision"""
    tzxy = tzxy0
    Npts =0
    for dist_th in np.linspace(resc,target,5):
        XT = X-tzxy
        ds,inds = KDTree(X_ref).query(XT,workers=20)
        keep = ds<dist_th
        X_ref_ = X_ref[inds[keep]]
        X_ = X[keep]
        tzxy = np.mean(X_-X_ref_,axis=0)
        #print(tzxy)
        Npts = np.sum(keep)
    return tzxy,Npts
def get_best_translation_points(X,X_ref,resc=5,target=3,constr=None,return_counts=False):
    XFF = np.concatenate([X,X_ref])
    X = X-np.min(XFF,axis=0)
    X_ref = X_ref-np.min(XFF,axis=0)
    if constr is not None:
        Xm = np.max([np.min(X,axis=0),np.min(X_ref,axis=0)],axis=0)-constr
        XM = np.min([np.max(X,axis=0),np.max(X_ref,axis=0)],axis=0)+constr
        keep = np.all((X<=XM)&(X>=Xm),axis=-1)
        X = X[keep]
        keep = np.all((X_ref<=XM)&(X_ref>=Xm),axis=-1)
        X_ref = X_ref[keep]
        XFF = np.concatenate([X,X_ref])
        X = X-np.min(XFF,axis=0)
        X_ref = X_ref-np.min(XFF,axis=0)
    
    im = get_im_from_Xh(X,resc=resc)
    im_ref = get_im_from_Xh(X_ref,resc=resc)
    
    from scipy.signal import fftconvolve
    im_cor = fftconvolve(im,im_ref[::-1,::-1,::-1])
    #plt.imshow(np.max(im_cor,0))
    tzxy = np.array(np.unravel_index(np.argmax(im_cor),im_cor.shape))-im_ref.shape+1
    tzxy = tzxy*resc
    Npts=0
    tzxy,Npts = get_Xtzxy(X,X_ref,tzxy,resc=resc,target=target)
    if return_counts:
        return tzxy,Npts
    return tzxy
def calc_color_matrix(x,y,order=2):
    """This gives a quadratic color transformation (in matrix form)
    x is Nx3 vector of positions in the reference channel (typically cy5)
    y is the Nx3 vector of positions in another channel (i.e. cy7)
    return m_ a 3x7 matrix which when multipled with x,x**2,1 returns y-x
    This m_ is indended to be used with apply_colorcor
    """ 
    x_ = np.array(y)# ref zxy
    y_ = np.array(x)-x_# dif zxy
    # get a list of exponents
    exps = []
    for p in range(order+1):
        for i in range(p+1):
            for j in range(p+1):
                if i+j<=p:
                    exps.append([i,j,p-i-j])
    # construct A matrix
    A = np.zeros([len(x_),len(exps)])
    for iA,(ix,iy,iz) in enumerate(exps):
        s = (x_[:,0]**ix*x_[:,1]**iy*x_[:,2]**iz)
        A[:,iA]=s
    m_ = [np.linalg.lstsq(A, y_[:,iy])[0] for iy in range(len(x_[0]))]
    m_=np.array(m_)
    return m_
def apply_colorcor(x,m=None):
    """This applies chromatic abberation correction to order 2
    x is a Nx3 vector of positions (typically 750(-->647))
    m is a matrix computed by function calc_color_matrix
    y is the corrected vector in another channel"""
    if m is None:
        return x
    exps = []
    order_max=10
    for p in range(order_max+1):
        for i in range(p+1):
            for j in range(p+1):
                if i+j<=p:
                    exps.append([i,j,p-i-j])
    #find the order
    mx,my = m.shape
    order = int((my-1)/mx)
    assert(my<len(exps))
    x_ = np.array(x)
    # construct A matrix
    exps = exps[:my]
    A = np.zeros([len(x_),len(exps)])
    for iA,(ix,iy,iz) in enumerate(exps):
        s = (x_[:,0]**ix*x_[:,1]**iy*x_[:,2]**iz)
        A[:,iA]=s
    diff = [np.dot(A,m_) for m_ in m]
    return x_+np.array(diff).T
def get_tzxy_plus_minus(obj_Xh_plus,obj_Xh_minus,obj_ref_Xh_plus,obj_ref_Xh_minus,resc=5,th=0):
    tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus = np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),0,0
    if (len(obj_Xh_plus)>0) and (len(obj_ref_Xh_plus)>0):
        X = obj_Xh_plus[obj_Xh_plus[:,-1]>th][:,:3]
        X_ref = obj_ref_Xh_plus[obj_ref_Xh_plus[:,-1]>th][:,:3]#obj_ref_Xh_plus[:,:3]
        tzxy_plus,N_plus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if (len(obj_Xh_minus)>0) and (len(obj_ref_Xh_minus)>0):
        X = obj_Xh_minus[obj_Xh_minus[:,-1]>th][:,:3]
        X_ref = obj_ref_Xh_minus[obj_ref_Xh_minus[:,-1]>th][:,:3]#obj_ref_Xh_plus[:,:3]
        tzxy_minus,N_minus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if np.max(np.abs(tzxy_minus-tzxy_plus))<=2:
        tzxyf = -(tzxy_plus*N_plus+tzxy_minus*N_minus)/(N_plus+N_minus)
    else:
        tzxyf = -[tzxy_plus,tzxy_minus][np.argmax([N_plus,N_minus])]
    return [tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus]    
def get_best_translation_pointsT(fov,htag,htagref,fov_folder,fov_folder_ref,set_='',resc=5,th=0):
    fl_feats = fov_folder /  f'{fov}--{htag}--dapiFeatures.npz'
    fl_feats_ref = fov_folder_ref / f'{fov}--{htagref}--dapiFeatures.npz'
    obj_Xh_plus,obj_Xh_minus = np.load(fl_feats)['Xh_plus'],np.load(fl_feats)['Xh_minus']
    obj_ref_Xh_plus,obj_ref_Xh_minus = np.load(fl_feats_ref)['Xh_plus'],np.load(fl_feats_ref)['Xh_minus']
    tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus = get_tzxy_plus_minus(obj_Xh_plus,obj_Xh_minus,obj_ref_Xh_plus,obj_ref_Xh_minus,resc=resc,th=th)
    return tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus
def read_im(fl,return_pos=False,ncols=4):
    import dask.array as da
    data = os.path.dirname(fl) / os.path.basename(fl).split('_')[-1].split('.')[0]+r'\data'
    im = da.from_zarr(fl,component=data)
    im = im[1:]
    im = im.reshape([-1,ncols,im.shape[-2],im.shape[-1]])
    im = im.swapaxes(0,1)
    
    im=im.astype(np.float32)
    im=im*im
    if return_pos is False:
        return im
    else:
        fl_xml = fl.replace('.zarr','.xml')
        x,y = [eval(ln.split('>')[1].split('<')[0]) for ln in open(fl_xml) if 'stage_position' in ln][0]
        return im,x,y
def get_icodesV3(dec,nmin_bits=3,iH=-3,save=False,make_unique=False):
    import time
    start = time.time()
    lens = dec.lens
    res_unfolder = dec.res_unfolder
    Mlen = np.max(lens)
    print("Calculating indexes within cluster...")
    res_is = np.tile(np.arange(Mlen), len(lens))
    res_is = res_is[res_is < np.repeat(lens, Mlen)]
    print("Calculating index of molecule...")
    ires = np.repeat(np.arange(len(lens)), lens)
    #r0 = np.array([r[0] for r in res for r_ in r])
    print("Calculating index of first molecule...")
    r0i = np.concatenate([[0],np.cumsum(lens)])[:-1]
    r0 = res_unfolder[np.repeat(r0i, lens)]
    print("Total time unfolded molecules:",time.time()-start)
    import gc
    gc.collect()
    ### torch
    ires = torch.from_numpy(ires.astype(np.int64))
    res_unfolder = torch.from_numpy(res_unfolder.astype(np.int64))
    res_is = torch.from_numpy(res_is.astype(np.int64))
    
    import time
    start = time.time()
    print("Computing score...")
    scoreF = torch.from_numpy(dec.XH[:,iH])[res_unfolder]
    print("Total time computing score:",time.time()-start)
    
    
    ### organize molecules in blocks for each cluster
    def get_asort_scores():
        val = torch.max(scoreF)+2
        scoreClu = torch.zeros([len(lens),Mlen],dtype=torch.float64)+val
        scoreClu[ires,res_is]=scoreF
        asort = scoreClu.argsort(-1)
        scoreClu = torch.gather(scoreClu,dim=-1,index=asort)
        scoresF2 = scoreClu[scoreClu<val-1]
        return asort,scoresF2
    def get_reorder(x,val=-1):
        if type(x) is not torch.Tensor:
            x = torch.from_numpy(np.array(x))
        xClu = torch.zeros([len(lens),Mlen],dtype=x.dtype)+val
        xClu[ires,res_is] = x
        xClu = torch.gather(xClu,dim=-1,index=asort)
        xf = xClu[xClu>val]
        return xf
    
    
    import time
    start = time.time()
    import gc
    gc.collect()
    print("Computing sorting...")
    asort,scoresF2 = get_asort_scores()
    res_unfolder2 = get_reorder(res_unfolder,val=-1)
    del asort
    del scoreF
    import gc
    gc.collect()
    print("Total time sorting molecules by score:",time.time()-start)
    
    
    
    import time
    start = time.time()
    print("Finding best bits per molecules...")
    
    Rs = dec.XH[:,-1].astype(np.int64)
    Rs = torch.from_numpy(Rs)
    Rs_U = Rs[res_unfolder2]
    nregs,nbits = dec.codes_01.shape
    score_bits = torch.zeros([len(lens),nbits],dtype=scoresF2.dtype)-1
    score_bits[ires,Rs_U]=scoresF2
    
    
    codes_lib = torch.from_numpy(np.array(dec.codes__))
    
    
    codes_lib_01 = torch.zeros([len(codes_lib),nbits],dtype=score_bits.dtype)
    for icd,cd in enumerate(codes_lib):
        codes_lib_01[icd,cd]=1
    codes_lib_01 = codes_lib_01/torch.norm(codes_lib_01,dim=-1)[:,np.newaxis]
    print("Finding best code...")
    batch = 10000
    icodes_best = torch.zeros(len(score_bits),dtype=torch.int64)
    dists_best = torch.zeros(len(score_bits),dtype=torch.float32)
    from tqdm import tqdm
    for i in tqdm(range((len(score_bits)//batch)+1)):
        score_bits_ = score_bits[i*batch:(i+1)*batch]
        if len(score_bits_)>0:
            score_bits__ = score_bits_.clone()
            score_bits__[score_bits__==-1]=0
            score_bits__ = score_bits__/torch.norm(score_bits__,dim=-1)[:,np.newaxis]
            Mul = torch.matmul(score_bits__,codes_lib_01.T)
            max_ = torch.max(Mul,dim=-1)
            icodes_best[i*batch:(i+1)*batch] = max_.indices
            dists_best[i*batch:(i+1)*batch] = 2-2*max_.values
    
    
    keep_all_bits = torch.sum(score_bits.gather(1,codes_lib[icodes_best])>=0,-1)>=nmin_bits
    dists_best_ = dists_best[keep_all_bits]
    score_bits = score_bits[keep_all_bits]
    icodes_best_ = icodes_best[keep_all_bits]
    icodesN=icodes_best_
    
    indexMols_ = torch.zeros([len(lens),nbits],dtype=res_unfolder2.dtype)-1
    indexMols_[ires,Rs_U]=res_unfolder2
    indexMols_ = indexMols_[keep_all_bits]
    indexMols_ = indexMols_.gather(1,codes_lib[icodes_best_])
    
    # make unique
    dec.dist_best = dists_best_.numpy()
    if make_unique:
        indexMols_,rinvMols = get_unique_ordered(indexMols_)
        icodesN = icodesN[rinvMols]
        dec.dist_best = dists_best_[rinvMols].numpy()
    XH = torch.from_numpy(dec.XH)
    XH_pruned = XH[indexMols_]
    XH_pruned[indexMols_==-1]=np.nan
    
    
    dec.XH_pruned=XH_pruned.numpy()
    dec.icodesN=icodesN.numpy()
    if save:
        np.savez_compressed(dec.decoded_fl,XH_pruned=dec.XH_pruned,icodesN=dec.icodesN,gns_names = np.array(dec.gns_names),dist_best=dec.dist_best)
    print("Total time best bits per molecule:",time.time()-start)
    import gc
    gc.collect()

def get_iH(fld): return int(os.path.basename(fld).split('_')[0][1:])
    
def get_XH(self,fov,ncols=3,th_h=0,medH_fl=None,color_fl=None):
    
    save_folder = self.save_folder
    drift_fl = save_folder / f'driftNew_{fov}--.pkl'
    drifts,all_flds,fov,fl_ref = pickle.load(open(drift_fl,'rb'))
    self.drifts,self.all_flds,self.fov,self.fl_ref = drifts,all_flds,fov,fl_ref
    
    XH = []
    for iH in tqdm(np.arange(len(all_flds))):
        fld = all_flds[iH]
        for icol in range(ncols):
            tag = os.path.basename(fld)
            save_fl = self.fov_folder / f'{fov.split('.')[0]}--{tag}--col{icol}__Xhfits.npz'
            Xh = np.load(save_fl,allow_pickle=True)['Xh']
            ### get drift
            tzxy = drifts[iH][0]
            ### get bit
            ih = get_iH(fld) 
            bit = (ih-1)*ncols+icol

            if len(Xh.shape):
                if medH_fl is not None:
                    ### color corection
                    medHs = np.load(medH_fl)['medHs']
                    Xh[:,-1]=Xh[:,-1]/medHs[bit]

            if len(Xh.shape):
                if len(Xh):
                    Xh = Xh[Xh[:,-1]>th_h]
                if len(Xh):
                    
                    icolR = np.array([[icol,bit]]*len(Xh))
                    
                    ### chromatic abberation correction
                    if color_fl is not None:
                        ms = np.load(color_fl,allow_pickle=True)
                        Xh[:,[0,1,2]] = apply_colorcor(Xh[:,[0,1,2]],ms[icol])
                    Xh[:,:3]+=tzxy# drift correction
                    XH_ = np.concatenate([Xh,icolR],axis=-1)
                    XH.extend(XH_)
    self.XH = np.array(XH)
def finished_fitting(fov_folder,fov,htags):
    is_good = []
    for htag in htags:
        fl_feats  = fov_folder / f"{fov}--{htag}--dapiFeatures.npz"
        is_good.append(fl_feats.exists())
    return np.all(is_good)
from scipy.spatial import KDTree
def set_scoreA(dec):
    score_ref = dec.score_ref
    score = dec.score
    from scipy.spatial import KDTree
    scoreA = np.zeros(len(score))
    for iS in range(score.shape[-1]):
        dist_,inds_ = KDTree(score_ref[:,[iS]]).query(score[:,[iS]],workers=20)
        scoreA+=np.log((inds_+1))-np.log(len(score_ref))
    dec.scoreA = scoreA
def get_intersV2(self,nmin_bits=3,dinstance_th=2):
    """Get an initial intersection of points within <dinstance_th>"""

    XH = self.XH
    Xs = XH[:,:3]
    Ts = KDTree(Xs)
    res = Ts.query_ball_point(Xs,dinstance_th,workers=20)
    print("Calculating lengths of clusters...")
    lens = np.array(list(map(len,res)))
    Mlen = np.max(lens)
    print("Unfolding indexes...")
    res_unfolder = np.concatenate(res)
    
    self.res_unfolder=res_unfolder
    self.lens=lens
    
    lens =self.lens
    self.res_unfolder = self.res_unfolder[np.repeat(lens, lens)>=nmin_bits]
    self.lens = self.lens[lens>=nmin_bits]
import time
def get_score(dec):
    H = np.nanmedian(dec.XH_pruned[...,-3],axis=1)
    n1bits = dec.XH_pruned.shape[1]
    from itertools import combinations
    combs = np.array(list(combinations(np.arange(n1bits),2)))
    X = dec.XH_pruned[:,:,:3].astype(np.float32)
    D = np.nanmean(np.linalg.norm(X[:,combs][:,:,0]-X[:,combs][:,:,1],axis=-1),axis=1)
    db = dec.dist_best
    score = np.array([H,-D,-db]).T
    #score = np.sort(score,axis=1)
    #score_ref = np.sort(score,axis=0)
    dec.score = score
    return score
class dummy():
    def __init__(self):
        pass
def load_library(self,lib_fl = r'Z:\DCBBL1_3_2_2023\MERFISH_Analysis\codebook_0_New_DCBB-300_MERFISH_encoding_2_21_2023.csv'):
        code_txt = np.array([ln.replace('\n','').split(',') for ln in open(lib_fl,'r') if ',' in ln])
        gns = code_txt[1:,0]
        code_01 = code_txt[1:,2:].astype(int)
        codes = np.array([np.where(cd)[0] for cd in code_01])
        codes_ = [list(np.sort(cd)) for cd in codes]
        nbits = np.max(codes)+1

        codes__ = codes_
        gns__ = list(gns)
        bad_gns = np.array(['blank' in e for e in gns__])
        good_gns = np.where(~bad_gns)[0]
        bad_gns = np.where(bad_gns)[0]

        
        
        self.lib_fl = lib_fl ### name of coding library
        self.nbits = nbits ### number of bits
        self.gns_names = gns__  ### names of genes and blank codes
        self.bad_gns = bad_gns ### indices of the blank codes
        self.good_gns = good_gns ### indices of the good gene codes
        self.codes__ = codes__ ### final extended codes of form [bit1,bit2,bit3,bit4]
        self.codes_01 = code_01

        dic_bit_to_code = {}
        for icd,cd in enumerate(self.codes__): 
            for bit in cd:
                if bit not in dic_bit_to_code: dic_bit_to_code[bit]=[]
                dic_bit_to_code[bit].append(icd)
        self.dic_bit_to_code = dic_bit_to_code  ### a dictinary in which each bit is mapped to the inde of a code

def plot_statistics(dec,ngns=20):
    if hasattr(dec,'im_segm_'):
        ncells = len(np.unique(dec.im_segm_))-1
    else:
        ncells = 1
    icds,ncds = np.unique(dec.icodesN[dec.scoreA>dec.th],return_counts=True)
    good_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' not in gn.lower()]
    kp = np.isin(icds,good_igns)
    top_genes = list(np.array(dec.gns_names)[icds[kp][np.argsort(ncds[kp])[::-1]]][:ngns])
    top_genes = [str(gn) for gn in top_genes]
    print(top_genes)
    ncds = ncds/ncells
    plt.figure()
    plt.xlabel('Genes')
    plt.semilogy(icds[kp],ncds[kp],label='genes')
    plt.semilogy(icds[~kp],ncds[~kp],label='blank')
    plt.ylabel('Number of molecules in the fov')
    plt.title(str(np.round(np.mean(ncds[~kp])/np.mean(ncds[kp]),3)))
    plt.legend()
def plot_multigenes(self,genes=['Gad1','Sox9'],colors=['r','g','b','m','c','y','w'],smin=3,smax=10,viewer = None,
                    drift=[0,0,0],resc=[1,1,1]):
    icodesN,XH_pruned = self.icodesN,self.XH_pruned
    scoreA=self.scoreA
    th=self.th
    gns_names = list(self.gns_names)
    
    Xcms = np.nanmean(XH_pruned,axis=1)
    keep = scoreA>th
    X = (Xcms[:,:3][keep]-drift)/resc  
    H = scoreA[keep]
    H -= np.min(H)
    icodesf = icodesN[keep]
    size = smin+np.clip(H/np.max(H),0,1)*(smax-smin)
    
    if viewer is None:
        import napari
        viewer = napari.Viewer()
    for igene,gene in enumerate(genes):
        color= colors[igene%len(colors)]
        icode = gns_names.index(gene)
        is_code = icode==icodesf
        viewer.add_points(X[is_code][:,-2:],size=size[is_code],face_color=color,name=gene,border_width=0)

    return viewer

def fit(
    fov_folder:pathlib.Path,
    save_folder:pathlib.Path,
    helper_folder:pathlib.Path, 
    Nhybes:int=9,
    verbose:bool = False):
    fov = next(fov_folder.iterdir()).stem.split("--")[0]
    dec = dummy()
    dec.fov_folder = fov_folder
    dec.save_folder = save_folder
    dec.decoded_fl = dec.save_folder / f'decodedNew_{fov}.npz'
    if not dec.decoded_fl.exists():
        ### compute drift
        
        htags = [rf'H{i+1}_MER_set1' for i in np.arange(Nhybes)]
        if not finished_fitting(fov_folder,fov,htags):
            raise NameError(f"Cannot find all dapiFeautres files in {fov_folder} based on Nhybes={Nhybes}")
            #time.sleep(10)
        htagref = htags[0]
        ### compute drift for all MERFISH rounds
        drift_fl = save_folder / f'driftNew_{fov}--.pkl'
        dec.drift_fl = drift_fl
        if not dec.drift_fl.exists():
            newdrifts = []
            for htag in htags:
                drft = get_best_translation_pointsT(fov,htag,htagref,fov_folder,fov_folder ,set_='',resc=5,th=0)
                if verbose:
                    print(htag,drft)
                newdrifts.append(drft)
            pickle.dump([newdrifts,htags,fov,htagref],open(drift_fl,'wb'))

        ### load in fitted data
        get_XH(dec,fov,ncols=3,th_h=0,
                   color_fl=helper_folder / 'color_correction.pkl',
                   medH_fl=helper_folder / 'medHBRBB.npz')
        
        
        dec.ncols = 3
        lib_fl = helper_folder / r'codebook_BRBB_500Markergn.csv'
        load_library(dec,lib_fl)
        get_intersV2(dec,nmin_bits=3,dinstance_th=3)
        get_icodesV3(dec,nmin_bits=3,iH=-3)
        get_score(dec)
        
        scores_ref_fl = save_folder / 'scores_BRBB_th3.npy'
        if not os.path.exists(scores_ref_fl):
            score_ref = np.sort(dec.score,axis=0)
            dec.score_ref = score_ref
            np.save(scores_ref_fl,score_ref)
        else:
            dec.score_ref = np.load(scores_ref_fl)
            
        set_scoreA(dec)
        scoreA = dec.scoreA
        
        keep = dec.scoreA>-2
        print("Saving file:",dec.decoded_fl)
        np.savez_compressed(dec.decoded_fl,XH_pruned=dec.XH_pruned[keep],
                            icodesN=dec.icodesN[keep],
                            gns_names = np.array(dec.gns_names),
                            dist_best=dec.dist_best[keep],
                           scoreA = dec.scoreA[keep])
    else:
        dec.XH_pruned = np.load(dec.decoded_fl)['XH_pruned']
        dec.icodesN = np.load(dec.decoded_fl)['icodesN']
        dec.gns_names = np.load(dec.decoded_fl)['gns_names']
        dec.dist_best = np.load(dec.decoded_fl)['dist_best']
        dec.scoreA = np.load(dec.decoded_fl)['scoreA']
    if verbose:
        scoreA = dec.scoreA
        bad_inds = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' in gn]
        is_bad = np.isin(dec.icodesN,bad_inds)
        is_good_gn = ~is_bad
        th_min=-7.5
        plt.figure()
        kp = scoreA>th_min
        plt.hist(scoreA[(is_good_gn)&kp],density=True,bins=100,alpha=0.5,label='all genes')
        plt.hist(scoreA[(~is_good_gn)&kp],density=True,bins=100,alpha=0.5,label='blanks');
        plt.legend()
        plt.savefig(save_folder / "histplot.png")
        plt.close()
    
        dec.th=-1.
        dec.gns_names = np.array(dec.gns_names)
        plot_statistics(dec)
        plt.savefig(save_folder / "stats.png")
        plt.close()
    return dec

def _check_path(path) -> pathlib.Path:
    """
    Helper function to check if path is pathlib.Path and convert it if not.
    """
    return pathlib.Path(path) if not isinstance(path, pathlib.Path) else path

def main(fov_folder,save_folder,helper_folder,Nhybes=9,try_mode=False,verbose=False):
    fov_folder = _check_path(fov_folder)
    save_folder = _check_path(save_folder)
    helper_folder = _check_path(helper_folder)

    if try_mode:
        try:
            dec = fit(fov_folder,save_folder,helper_folder,Nhybes=Nhybes,verbose=verbose)
        except:
            print("Failed:",fov)
            dec = None
    else:
        dec = fit(fov_folder,save_folder,helper_folder,Nhybes=Nhybes,verbose=verbose)
    return dec

import sys
if __name__ == "__main__":
    fov = sys.argv[1]
    save_folder = sys.argv[2]
    helper_folder = sys.argv[3]
    dec = main(fov,save_folder,helper_folder,try_mode=False)