import pathlib as plib
import time
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf

import pyxu.experimental.xray as pxr
import pyxu.runtime as pxrt
import pyxu.opt.stop as pxst



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
    sod, sdd, v_shape, v_pitch, P = get_info(fpath)
    #flip image of each view
    
    # ROI
    #roi = 5
    #P = P[:, P.shape[1]//2 - roi: P.shape[1]//2 + roi, :]

    P = P[:, P.shape[1]//2, :]
    N_view, N_w = P.shape
    N_h = 1
    v_shape = (v_shape[0], 1, v_shape[2])
    # Compute TX position (1 projection). =====================================
    rx_h = rx_pitch[0] * dws * N_h  # detector height [mm]
    rx_w = rx_pitch[1] * dws * N_w  # detector width  [mm]
    
    shift_cor = -0.3
    tx_pos = np.r_[shift_cor, rx_h / 2, sod ]  # (3,)

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

    #fig = op.diagnostic_plot(slice(None, None, 600_000))
    #fig.savefig("./diagnostic_plot.png", dpi=300)

    '''
    plt.figure('tilt series')
    for i in range(P.shape[0]):
        plt.imshow(P[i,:,:])
        plt.pause(0.1)
        plt.clf()
    '''

    W = pxrt.Width.SINGLE
    with pxrt.Precision(W):
        '''
        V = np.ones(v_shape, dtype=W.value)
        V = V.reshape(-1)

        t_start = time.time()
        P = op.apply(V)
        t_stop = time.time()
        print("forward: ", t_stop - t_start)
        '''

        t_start = time.time()
        #V_bw = op.adjoint(P.reshape(-1))
        stop_crit = pxst.MaxIter(80)
        V_bw = op.pinv(P.reshape(-1), damp=8, kwargs_fit=dict(stop_crit=stop_crit))
        t_stop = time.time()
        print("backward: ", t_stop - t_start)

        plt.figure()
        plt.imshow(V_bw.reshape(v_shape)[:, v_shape[1]//2, :], cmap='gray')
        plt.show()          

        breakpoint()


    #function doing reconstruction with all this previous code taking as input the output of the function get_info, and also other parameters
    def setOperator(sod, sdd, v_shape, v_pitch, P, rx_pitch, dws, roi):
        P = P[:, P.shape[1]//2 - roi: P.shape[1]//2 + roi, :] #ROI taking horizontal slice of the projections
        N_view, N_h, N_w = P.shape
        # Compute TX position (1 projection). =====================================
        rx_h = rx_pitch[0] * dws * N_h  # detector height [mm]
        rx_w = rx_pitch[1] * dws * N_w  # detector width  [mm]
        tx_pos = np.r_[0, rx_h / 2, sod]  # (3,)

        # Compute RX position (1 projection). =====================================
        rx_ll = np.r_[-rx_w / 2, 0, sod - sdd]  # detector lower-left corner
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

        # Expand n/t_spec to include all rotations. ===============================
        angle = np.linspace(0, 2 * np.pi, num=N_view, endpoint=False)
        R = np.zeros(
            (N_view, 3, 3),
        )
        R[:, 0, 0] = np.cos(angle)
        R[:, 2, 2] = np.cos(angle)
        R[:, 2, 0] = np.sin(angle)
        R[:, 0, 2] = -np.sin(angle)
        R[:, 1, 1] = 1

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
        # Compute TX position (1 projection). =====================================

        return op

    #function with only path
