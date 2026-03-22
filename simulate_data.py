import numpy as np
import scipy
import scipy.ndimage
import os
import PIL
from PIL import Image
import h5py
from util_func import interpolate_projection, pkev2kvp, marBHC

def simulate_metal_artifact(imgCT, imgMetalList, CTpara, MARpara, output_dir, param, ray_trafo, FBPOper, pool):
    n_mask = imgMetalList.shape[2]
    # tissue composition
    MiuWater = MARpara['MiuWater']
    threshWater = MARpara['threshWater']
    threshBone = MARpara['threshBone']

    img = imgCT / 1000 * MiuWater + MiuWater
    gt_CT = img
    imgWater = np.zeros_like(img)
    imgBone = np.zeros_like(img)
    bwWater = img <= threshWater
    bwBone = img >= threshBone
    bwBoth = (1 - bwWater - bwBone) > 0.5
    imgWater[bwWater] = img[bwWater]
    imgBone[bwBone] = img[bwBone]
    imgBone[bwBoth] = (img[bwBoth] - threshWater) / (threshBone - threshWater) * img[bwBoth]
    imgWater[bwBoth] = img[bwBoth] - imgBone[bwBoth]

    # Synthesize non-metal poly CT
    Pwater_kev = ray_trafo(imgWater)
    Pbone_kev = ray_trafo(imgBone)
    Pwater_kev = Pwater_kev#/param.reso
    Pbone_kev = Pbone_kev#/param.reso

    NumofRou, NumofTheta = Pwater_kev.shape
    projkevAll = np.zeros((NumofRou, NumofTheta, 3))
    projkevAll[:, :, 0] = Pwater_kev
    projkevAll[:, :, 1] = Pbone_kev
    projkvp = pkev2kvp(projkevAll, MARpara['spectrum'], MARpara['energies'], MARpara['kev'], MARpara['MiuAll'])

    # Poisson noise
    scatterPhoton = 20
    temp = np.round(np.exp(-projkvp) * MARpara['photonNum'])
    temp = temp + scatterPhoton                               # simulate scattered photon
    ProjPhoton = np.random.poisson(temp)
    ProjPhoton[ProjPhoton==0] = 1
    projkvpNoise = -np.log(ProjPhoton / MARpara['photonNum'])

    # correction
    p1 = np.reshape(projkvpNoise, (NumofRou*NumofTheta, 1))
    p1BHC = np.matmul(np.concatenate([p1,  p1**2,  p1**3], axis=1), np.asarray(MARpara['paraBHC']))
    poly_sinogram = np.reshape(p1BHC, (NumofRou, NumofTheta))
    poly_CT = np.asarray(FBPOper(poly_sinogram))

    # ct_file = os.path.join(output_dir, 'gt.mat')
    # data_dict = {'image': gt_CT, 'poly_sinogram': poly_sinogram, 'poly_CT': poly_CT}
    # sio.savemat(ct_file, data_dict)

    ct_file = os.path.join(output_dir, 'gt.h5')
    f = h5py.File(ct_file, 'w')
    f.create_dataset('image', data=np.array(gt_CT, dtype=np.float32), compression="gzip")
    f.create_dataset('poly_sinogram', data=np.array(poly_sinogram, dtype=np.float32), compression="gzip")
    f.create_dataset('poly_CT', data=np.array(poly_CT, dtype=np.float32), compression="gzip")
    f.close()

    # Metal
    def generate_one_mask(idx):
        imgMetal = imgMetalList[:, :, idx].copy()
        imgMetal = np.array(
            Image.fromarray(imgMetal).resize((CTpara['imPixNum'], CTpara['imPixNum']), PIL.Image.BILINEAR))

        Pmetal_kev = np.asarray(ray_trafo(imgMetal))
        Pmetal_kev = Pmetal_kev #/ param.reso
        metal_trace = Pmetal_kev > 0
        Pmetal_kev = MARpara['metalAtten'] * Pmetal_kev

        # partial volume effect
        Pmetal_kev_bw = scipy.ndimage.binary_erosion(Pmetal_kev > 0, structure=np.ones((1, 3)))
        Pmetal_edge = np.logical_xor((Pmetal_kev > 0), Pmetal_kev_bw)
        Pmetal_kev[Pmetal_edge] = Pmetal_kev[Pmetal_edge] / 4

        # sinogram with metal
        projkevAllLocal = projkevAll.copy()
        projkevAllLocal[:, :, 2] = Pmetal_kev
        projkvpMetal = pkev2kvp(projkevAllLocal, MARpara['spectrum'], MARpara['energies'], MARpara['kev'],
                                MARpara['MiuAll'])
        temp = np.round(np.exp(-projkvpMetal) * MARpara['photonNum'])
        temp = temp + scatterPhoton
        ProjPhoton = np.random.poisson(temp)
        ProjPhoton[ProjPhoton == 0] = 1
        projkvpMetalNoise = -np.log(ProjPhoton / MARpara['photonNum'])

        # correction
        p1 = np.reshape(projkvpMetalNoise, (NumofRou * NumofTheta, 1))
        p1BHC = np.matmul(np.concatenate([p1, p1 ** 2, p1 ** 3], axis=1), np.asarray(MARpara['paraBHC']))
        ma_sinogram = np.reshape(p1BHC, (NumofRou, NumofTheta))
        LI_sinogram = interpolate_projection(ma_sinogram, metal_trace)
        ma_CT = np.asarray(FBPOper(ma_sinogram))
        LI_CT = np.asarray(FBPOper(LI_sinogram))
        #Sgt = np.asarray(ray_trafo(gt_CT))

        # print('ma_CT {}, LI_CT {}, poly_CT {}, gt_ct {}'.format(ma_CT.mean(), LI_CT.mean(), poly_CT.mean(), gt_CT.mean()))
        # print(
        #     'mas {}, LIs {}, polys {}, gts {}'.format(ma_sinogram.mean(), LI_sinogram.mean(), poly_sinogram.mean(), Sgt.mean()))
        ##save
        # ct_file = os.path.join(output_dir, str(idx) + '.mat')
        # data_dict = {'ma_CT': ma_CT,'LI_CT': LI_CT, 'ma_sinogram': ma_sinogram, 'LI_sinogram':LI_sinogram, 'metal_trace': metal_trace}
        # sio.savemat(ct_file, data_dict)

        imBHC,projBHC = marBHC(ma_sinogram, imgMetal, ray_trafo, FBPOper, param)

        ct_file = os.path.join(output_dir, str(idx) + '.h5')
        f = h5py.File(ct_file, 'w')
        f.create_dataset('ma_CT', data=np.array(ma_CT, dtype=np.float32), compression="gzip")
        f.create_dataset('LI_CT', data=np.array(LI_CT, dtype=np.float32), compression="gzip")
        f.create_dataset('ma_sinogram', data=np.array(ma_sinogram, dtype=np.float32), compression="gzip")
        f.create_dataset('LI_sinogram', data=np.array(LI_sinogram, dtype=np.float32), compression="gzip")
        f.create_dataset('BHC_sinogram', data=np.array(projBHC, dtype=np.float32), compression="gzip")
        f.create_dataset('BHC_CT', data=np.array(imBHC, dtype=np.float32), compression="gzip")
        #f.create_dataset('gt_sinogram', data=np.array(Sgt, dtype=np.float32), compression="gzip")
        f.create_dataset('metal_trace', data=np.array(metal_trace, dtype=np.uint8), compression="gzip")
        f.close()
    pool.map(generate_one_mask, np.arange(0, n_mask))
    #for idx in range(n_mask):
    #    generate_one_mask(idx)
    return
