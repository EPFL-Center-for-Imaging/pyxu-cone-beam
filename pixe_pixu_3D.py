import pathlib as plib
import time
import xml.etree.ElementTree as ET
import numpy as cp
import matplotlib.pyplot as plt
#import numpy as np
import tifffile as tf

import pyxu.experimental.xray as pxr
import pyxu.runtime as pxrt
import pyxu.opt.stop as pxst

from pyxu.operator import Gradient, SquaredL2Norm, L1Norm, PositiveL1Norm, L21Norm
from pyxu.opt.solver import PGD, CV
from pyxu.abc import ProxFunc



def get_info(path: plib.Path):
    # Parameters
    # ----------
    # path: plib.Path
    #     Folder where .xml file and raw data is located.
    #
    # Returns
    # -------
    # sod: float
    #     Src-object distance.
    # sdd: float
    #     Src-detector distance.
    # v_shape: tuple[int]
    #     (x, y, z) pixel count.
    # v_pitch: tuple[float]
    #     (x, y, z) pixel pitch.
    # P: cp.ndarray[float32]
    #     (N_view, N_h, N_w) projections.
    meta_path = path / "unireconstruction.xml"
    meta = ET.parse(meta_path).getroot()

    # Extract sod/sdd.
    geom = meta.find(".//geometry")
    sod = float(geom.attrib["sod"])
    sdd = float(geom.attrib["sdd"])

    # Extract pitch/shape.
    vol = meta.find(".//volume")
    v_shape = tuple(map(int, (vol.find("./size").attrib[_] for _ in ["X", "Y", "Z"])))
    v_pitch = tuple(map(float, (vol.find("./voxelSize").attrib[_] for _ in ["X", "Y", "Z"])))
    # Extract projections (N_view, y, z)
    P_folder = path / "Proj"
    P_files = sorted(P_folder.glob("*.tif"))
    tf_meta = tf.TiffFile(P_files[0])
    assert tf_meta.pages[0].axes == "YX"  # all files are (row, column)-ordered.

    P = cp.stack([cp.array(tf.imread(f)) for f in P_files], axis=0)
    P = P.astype(cp.single) / cp.iinfo(P.dtype).max

    return sod, sdd, v_shape, v_pitch, cp.array(P)

def setOperator(sod, sdd, v_shape, v_pitch, P, rx_pitch, dws, roi='False', shift_cor=0, center_slice='False'):
    '''
    Parameters
    ----------
    sod: float
        Src-object distance.
    sdd: float
        Src-detector distance.
    v_shape: tuple[int]
        (x, y, z) pixel count.
    v_pitch: tuple[float]
        (x, y, z) pixel pitch.
    P: cp.ndarray[float32]
        (N_view, N_h, N_w) projections.
    rx_pitch: tuple[float]
        (x, y) detector pitch.
    dws: int
        Down-sampling rate.
    roi: int
        Region of interest.
    shift_cor: float
        Shift correction value for center of rotation.
    center_slice: bool
        If True, only center slice of projections is used for reconstruction.
    Returns
    -------
    op: pxr.XRayTransform
        Operator.
    P: cp.ndarray[float32]
        (N_view, N_h, N_w) projections.
    v_shape: tuple[int]
        (x, y, z) pixel count.
    '''
    
    #flip image of each view
    
    if roi != 'False':
        P = P[:, P.shape[1]//2 - roi: P.shape[1]//2 + roi, :]
        v_shape = (v_shape[0], 2*roi, v_shape[2])

    N_view, N_h, N_w = P.shape
    
    if center_slice == 'True':
        P = P[:, P.shape[1]//2, :]
        N_view, N_w = P.shape
        N_h = 1
        v_shape = (v_shape[0], 1, v_shape[2])

    # Compute TX position (1 projection). =====================================
    rx_h = rx_pitch[0] * dws * N_h  # detector height [mm]
    rx_w = rx_pitch[1] * dws * N_w  # detector width  [mm]

    # normalize data    
    P  = -cp.log(P/P.max())
    
    tx_pos = cp.r_[shift_cor, rx_h / 2, sod]  # (3,)

    # Compute RX position (1 projection). =====================================
    rx_ll = cp.r_[-rx_w / 2 + shift_cor, 0, sod - sdd ]  # detector lower-left corner
    
    Y, X, Z = cp.meshgrid(
        (0.5 * rx_pitch[0] * dws) + cp.arange(N_h) * (rx_pitch[0] * dws),  # height (Y-axis)
        (0.5 * rx_pitch[1] * dws) + cp.arange(N_w) * (rx_pitch[1] * dws),  # width  (X-axis)
        cp.r_[0],  # depth (Z-axis)
        indexing="ij",
    )
    
    rx_pos = rx_ll + cp.concatenate([X, Y, Z], axis=-1)  # (N_h, N_w, 3)
    # Compute VOL position. ===================================================
    v_dim = cp.r_[v_shape] * cp.r_[v_pitch]  # volume XYZ dimensions [mm]      
    v_center = cp.r_[0, rx_h / 2, 0]  # volume center                          
    v_ll = v_center - v_dim / 2  # volume lower-left corner.                   

    # Compute ray parameterizations (1 projection). ===========================
    n_spec = cp.reshape(rx_pos - tx_pos, (-1, 3))  # (N_h * N_w, 3)       
    t_spec = cp.broadcast_to(tx_pos, n_spec.shape)  # (N_h * N_w, 3)           
    #t_spec = cp.reshape(rx_pos, (-1,3)) # (N_h * N_w, 3)           

    # Expand n/t_spec to include all rotations. ===============================
    angle = -cp.linspace(0, 2 * cp.pi, num=N_view, endpoint=False) #+ cp.pi
    R = cp.zeros(
        (N_view, 3, 3),
    )
    R[:, 0, 0] = cp.cos(angle)
    R[:, 2, 2] = cp.cos(angle)
    R[:, 2, 0] = cp.sin(angle)
    R[:, 0, 2] = -cp.sin(angle)
    R[:, 1, 1] = 1

    #shift t_spec third dimension to correction center of rotation
    n_spec = (n_spec @ R.transpose(0, 2, 1)).reshape(-1, 3)  # (N_view * N_h * N_w, 3)
    t_spec = (t_spec @ R.transpose(0, 2, 1)).reshape(-1, 3)  # (N_view * N_h * N_w, 3)

    op = pxr.XRayTransform.init(
        arg_shape=v_shape,
        origin=v_ll,
        pitch=v_pitch,
        method="ray-trace",
        n_spec=n_spec,
        t_spec=t_spec,
    )

    return op, P, v_shape

def reconstruct(fpath, dataset, dws, rx_pitch, method = 'pinv', roi='False', shift_cor=0., center_slice='False', max_iter=40, damp=0.1):
    '''
    Parameters
    ----------
    fpath: plib.Path
        Folder where .xml file and raw data is located.
    dataset: str
        Name of dataset.
    dws: int
        Down-sampling rate.
    rx_pitch: tuple[float]
        (x, y) detector pitch.
    method: str
        Reconstruction method : 'pinv', 'tv'.
    roi: int
        Region of interest.
    shift_cor: float
        Shift correction value for center of rotation.
    center_slice: bool
        If True, only center slice of projections is used for reconstruction.
    max_iter: int
        Maximum number of iterations.
    damp: float
        Damping factor.
    Returns
    -------
    V_bw: cp.ndarray[float32]
        (x, y, z) reconstruction.
    '''
    
    sod, sdd, v_shape, v_pitch, P = get_info(fpath)
    
    op, P, v_shape = setOperator(sod, sdd, v_shape, v_pitch, P, rx_pitch, dws, roi, shift_cor, center_slice)
    W = pxrt.Width.SINGLE
    with pxrt.Precision(W):
        t_start = time.time()
        if method == 'pinv':
            stop_crit = pxst.MaxIter(max_iter)
            V_bw = op.pinv(P.reshape(-1), damp=damp, kwargs_fit=dict(stop_crit=stop_crit))
            
        elif method == 'tv':
            stop_crit = pxst.MaxIter(max_iter)
            x0= cp.zeros(v_shape).ravel()

            class TVFunc(ProxFunc):
                r"""
                TODO
                """

                def __init__(self,
                            arg_shape,
                            isotropic: bool = True,
                            finite_diff_kwargs: dict = dict(),
                            prox_init_kwargs: dict = dict(show_progress=False),
                            prox_fit_kwargs: dict = dict(),
                            ):
                    r"""
                    Parameters
                    ----------
                    arg_shape: pyct.NDArrayShape
                        Shape of the input array.
                    isotropic: bool
                        Isotropic or anisotropic TV (Default: isotropic).
                    """
                    
                    N_dim = len(arg_shape)

                    super().__init__(shape=(1, arg_shape[0]*arg_shape[1]*arg_shape[2]))
                    
                    self._lipschitz = cp.inf 
                    self._arg_shape = arg_shape
                    
                    finite_diff_op = Gradient(arg_shape, sampling=[1, 1, 1]) #GPU=True if gpu is available /!\
                    
                    #finite_diff_op.estimate_lipschitz()
                    finite_diff_op.lipschitz= 3
                    self._finite_diff_op = finite_diff_op
                    
                    if isotropic:
                        self._norm = L21Norm(arg_shape=(N_dim, *arg_shape))
                    
                    #self._norm = L1Norm(dim=finite_diff_op.codim)
                    self._prox_init_kwargs = prox_init_kwargs
                    self._prox_fit_kwargs = prox_fit_kwargs

                def apply(self, arr):
                    return cp.array(self._norm(self._finite_diff_op(cp.array(arr))))

                def prox(self, arr, tau = 0.05):
                    ls = 1 / 2 * SquaredL2Norm(dim=arr.size).argshift(-cp.array(arr))
                    #ls.estimate_diff_lipschitz()
                    ls.diff_lipschitz = 1.0
                    slv = CV(f=ls, h=tau * self._norm, K=self._finite_diff_op, verbosity = 5)
                    slv.fit(x0=cp.array(arr.copy()), stop_crit = pxst.RelError(eps=1e-4, var="x", f=None, norm=2, satisfy_all=True) | pxst.MaxIter(30))
                    return cp.array(slv.solution().reshape(arr.shape))
                
            lambda_= damp
            tv_prior = TVFunc(arg_shape=v_shape)
            # Loss, smooth part of the posterior
            sigma = 1
            loss = (1/ (2 * sigma**2)) * SquaredL2Norm(dim=P.size).asloss(P.ravel()) * op
            smooth_posterior = loss
            #Convergence of the algorithm is not guaranteed if the Lipschitz constant is not known
            smooth_posterior.diff_lipschitz = 1e3 #smooth_posterior.estimate_diff_lipschitz() ??
            
            # Define the solver
            solver = PGD(f=smooth_posterior, g = lambda_ * tv_prior, show_progress=True, verbosity=1)
            
            # Call fit to trigger the solver
            stop_crit = pxst.RelError(eps=1e-3, var="x", f=None, norm=2, satisfy_all=True) | pxst.MaxIter(max_iter)

            # Launch the solver
            solver.fit(x0=x0, acceleration=True, stop_crit=stop_crit)

            recon_tv = solver.solution().squeeze()
            V_bw = recon_tv.reshape(v_shape)

        t_stop = time.time()
        
        return V_bw.reshape(v_shape)

def shift_correction(fpath, dataset, dws, rx_pitch, range_mm, step, metric = 'variance'):
    '''
    Parameters
    ----------
    fpath: plib.Path
        Folder where .xml file and raw data is located.
    dataset: str
        Name of dataset.
    dws: int
        Down-sampling rate.
    rx_pitch: tuple[float]
        (x, y) detector pitch.
    range: tuple[float]
        Range of shift correction values.
    step: float
        Step size of shift correction values.
    metric: str
        Metric to choose shift correction value : 'variance', 'fft'.

    Returns
    -------
    shift_corr: float
        Shift correction value for center of rotation.
    '''

    #find correct shift correction with plotting curve of sharpness if reconstructions for different shift corrections
    shift_corr = cp.arange(range_mm[0], range_mm[1], step)
    sharpness = cp.zeros(shift_corr.shape)

    if metric == 'fft':
        print('Center of rotation correction with method: ', metric)
        for i in range(len(shift_corr)):
            #sharpness in fourier space : mask to keep only high frequencies and then sum absolute values
            rec = reconstruct(fpath, dataset, dws, rx_pitch, method = 'pinv', roi='False', shift_cor=shift_corr[i], center_slice='True', max_iter=4, damp=0.05)
            rec = rec[:, rec.shape[1]//2, :]
            fourier_image = cp.fft.fftshift(cp.fft.fft2(rec))
            fourier_image = cp.abs(fourier_image)
            mask = cp.ones(fourier_image.shape)
            x, y = cp.ogrid[:fourier_image.shape[0], :fourier_image.shape[1]]
            mask = cp.sqrt((x - fourier_image.shape[0]//2)**2 + (y - fourier_image.shape[1]//2)**2) <= 150
            sharpness[i] = cp.sum(fourier_image*mask)

    elif metric == 'variance':
        print('Center of rotation correction with method: ', metric)
        for i in range(len(shift_corr)):
            #sharpness in image space : variance of image
            rec = reconstruct(fpath, dataset, dws, rx_pitch, method = 'pinv', roi='False', shift_cor=shift_corr[i], center_slice='True', max_iter=4, damp=0.05)
            rec = rec[:, rec.shape[1]//2, :]
            sharpness[i] = cp.var(rec)

    #plot sharpness curve
    plt.figure()
    plt.plot(shift_corr, sharpness)
    plt.xlabel('shift correction value')
    plt.ylabel('sharpness')
    plt.show()

    return shift_corr[cp.argmax(sharpness)]

if __name__ == "__main__":
    # All coordinates are in XYZ ordering.
    # Reference frame centered at base of rotating plate, with:
    # * XZ-plane horizontal (touching bottom of detector);
    # * Y-axis pointing up;
    # * Z-axis pointing towards TX.

    # A-priori knowledge of the datasets and machine. =========================
    # dataset, dws = "000-HR", 2  # folder, down-sampling rate
    dataset, dws = "bin2", 2
    rx_pitch = (127e-3, 127e-3)  # 0.127 [mm] detector/receiver pitch (height, width)

    # Load setup parameters from disk. ========================================
    froot = plib.Path("~/Downloads/M2EA99-000").expanduser()
    fpath = froot / dataset

    shiftcorr_value = -0.293 #precalculated shift correction value, below is code to calculate it
    #shiftcorr_value = shift_correction(fpath, dataset, dws, rx_pitch, range_mm=[-0.5, 0.1], step=0.01, metric = 'variance') #comment/uncomment to calculate shift correction value
    print('shift correction value: ', shiftcorr_value)

    # Reconstruct. ============================================================
    rec = reconstruct(fpath, dataset, dws, rx_pitch, method = 'tv', roi=70, shift_cor=shiftcorr_value, center_slice='True', max_iter=120, damp=0.009)
    
    # Plot. ===================================================================
    plt.figure()
    plt.imshow(rec[100:500, rec.shape[1]//2, 100:500], cmap='gray', vmax=0.07, vmin=0.0015) #vmax=0.1)
    plt.show()

    cp.save('recon_pixe_hor', rec[100:500, rec.shape[1]//2, 100:500])
    cp.save('recon_pixe_vert', rec[100:500, :, rec.shape[2]//2])
    cp.save('recon_pixe_sag', rec[rec.shape[0]//2, :, :])

    breakpoint()
