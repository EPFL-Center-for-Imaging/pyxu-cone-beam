import pathlib as plib
import time
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf

import pyxu.experimental.xray as pxr
import pyxu.runtime as pxrt
import pyxu.opt.stop as pxst

from pyxu.operator import Gradient, SquaredL2Norm, L1Norm, PositiveL1Norm
from pyxu.opt.solver import PGD


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
    # P: np.ndarray[float32]
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
    P = np.stack([tf.imread(f) for f in P_files], axis=0)
    P = P.astype(np.single) / np.iinfo(P.dtype).max

    return sod, sdd, v_shape, v_pitch, P

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
    P: np.ndarray[float32]
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
    P: np.ndarray[float32]
        (N_view, N_h, N_w) projections.
    v_shape: tuple[int]
        (x, y, z) pixel count.

        
    '''
    
    #flip image of each view
    if roi != 'False':
        P = P[:, P.shape[1]//2 - roi: P.shape[1]//2 + roi, :]

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
    P  = -np.log(P/P.max())
    
    tx_pos = np.r_[shift_cor, rx_h / 2, sod]  # (3,)

    # Compute RX position (1 projection). =====================================
    rx_ll = np.r_[-rx_w / 2 + shift_cor, 0, sod - sdd ]  # detector lower-left corner
    
    Y, X, Z = np.meshgrid(
        (0.5 * rx_pitch[0] * dws) + np.arange(N_h) * (rx_pitch[0] * dws),  # height (Y-axis)
        (0.5 * rx_pitch[1] * dws) + np.arange(N_w) * (rx_pitch[1] * dws),  # width  (X-axis)
        np.r_[0],  # depth (Z-axis)
        indexing="ij",
    )
    
    rx_pos = rx_ll + np.concatenate([X, Y, Z], axis=-1)  # (N_h, N_w, 3)
    # Compute VOL position. ===================================================
    v_dim = np.r_[v_shape] * np.r_[v_pitch]  # volume XYZ dimensions [mm]      
    v_center = np.r_[0, rx_h / 2, 0]  # volume center                          
    v_ll = v_center - v_dim / 2  # volume lower-left corner.                   

    # Compute ray parameterizations (1 projection). ===========================
    n_spec = np.reshape(rx_pos - tx_pos, (-1, 3))  # (N_h * N_w, 3)       
    t_spec = np.broadcast_to(tx_pos, n_spec.shape)  # (N_h * N_w, 3)           
    #t_spec = np.reshape(rx_pos, (-1,3)) # (N_h * N_w, 3)           



    # Expand n/t_spec to include all rotations. ===============================
    angle = -np.linspace(0, 2 * np.pi, num=N_view, endpoint=False) #+ np.pi
    R = np.zeros(
        (N_view, 3, 3),
    )
    R[:, 0, 0] = np.cos(angle)
    R[:, 2, 2] = np.cos(angle)
    R[:, 2, 0] = np.sin(angle)
    R[:, 0, 2] = -np.sin(angle)
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
    V_bw: np.ndarray[float32]
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
            x0= np.zeros(v_shape).ravel()
            grad = Gradient(arg_shape=v_shape)
            lambda_= damp
            huber_norm = L1Norm(grad.shape[0]).moreau_envelope(0.01)  # We smooth the L1 norm to facilitate optimisation
            tv_prior = lambda_ * huber_norm * grad

            # Loss
            sigma = 1
            loss = (1/ (2 * sigma**2)) * SquaredL2Norm(dim=P.size).asloss(P.ravel()) * op

            # Smooth part of the posterior
            smooth_posterior = loss + tv_prior
            smooth_posterior.diff_lipschitz = 3e3 #smooth_posterior.estimate_diff_lipschitz()

            # Define the solver
            solver = PGD(f=smooth_posterior, show_progress=True, verbosity=1)

            # Call fit to trigger the solver
            stop_crit = pxst.RelError(eps=1e-3, var="x", f=None, norm=2, satisfy_all=True) | pxst.MaxIter(max_iter)
            solver.fit(x0=x0, acceleration=True, stop_crit=stop_crit)
            recon_tv = solver.solution().squeeze()
            V_bw = recon_tv.reshape(v_shape)

        t_stop = time.time()
        #print("backward: ", t_stop - t_start)
        
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
    shift_corr = np.arange(range_mm[0], range_mm[1], step)
    sharpness = np.zeros(shift_corr.shape)

    if metric == 'fft':
        print('Center of rotation correction with method: ', metric)
        for i in range(len(shift_corr)):
            #sharpness in fourier space : mask to keep only high frequencies and then sum absolute values
            rec = reconstruct(fpath, dataset, dws, rx_pitch, method = 'pinv', roi='False', shift_cor=shift_corr[i], center_slice='True', max_iter=4, damp=0.05)
            rec = rec[:, rec.shape[1]//2, :]
            fourier_image = np.fft.fftshift(np.fft.fft2(rec))
            fourier_image = np.abs(fourier_image)
            mask = np.ones(fourier_image.shape)
            x, y = np.ogrid[:fourier_image.shape[0], :fourier_image.shape[1]]
            mask = np.sqrt((x - fourier_image.shape[0]//2)**2 + (y - fourier_image.shape[1]//2)**2) <= 150
            sharpness[i] = np.sum(fourier_image*mask)
    elif metric == 'variance':
        print('Center of rotation correction with method: ', metric)
        for i in range(len(shift_corr)):
            print(i)
            #sharpness in image space : variance of image
            rec = reconstruct(fpath, dataset, dws, rx_pitch, method = 'pinv', roi='False', shift_cor=shift_corr[i], center_slice='True', max_iter=4, damp=0.05)
            rec = rec[:, rec.shape[1]//2, :]
            sharpness[i] = np.var(rec)

    plt.figure()
    plt.plot(shift_corr, sharpness)
    plt.xlabel('shift correction value')
    plt.ylabel('sharpness')
    plt.show()

    return shift_corr[np.argmax(sharpness)]

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

    shiftcorr_value = -0.293 #precalculated shift correction value
    #shiftcorr_value = shift_correction(fpath, dataset, dws, rx_pitch, range_mm=[-0.5, 0.1], step=0.01, metric = 'variance') #comment/uncomment to calculate shift correction value
    print('shift correction value: ', shiftcorr_value)
    rec = reconstruct(fpath, dataset, dws, rx_pitch, method = 'pinv', roi='False', shift_cor=shiftcorr_value, center_slice='True', max_iter=100, damp=3)
    
    plt.figure()
    plt.imshow(rec[100:500, rec.shape[1]//2, 100:500], cmap='gray', vmax=0.1, vmin=0)
    plt.show()   


    breakpoint()
