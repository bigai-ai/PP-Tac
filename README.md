# PP-Tac: Paper Picking Using Tactile Feedback in Dexterous Robotic Hands

[Pei Lin<sup>*</sup>](http://peilin-666.github.io/)<sup>1,2</sup>,
[YuZhe Huang<sup>*</sup>]<sup>1,3</sup>,
[Wanlin Li<sup>*</sup>](https://mrliwanlin.github.io/)<sup>1</sup>,
[Jianpeng Ma]<sup>1</sup>,
[Chenxi Xiao<sup>†</sup>](https://scholar.google.com/citations?hl=zh-CN&user=Qhiy3doAAAAJ)<sup>2</sup>,
[Ziyuan Jiao<sup>†</sup>](https://scholar.google.com/citations?user=YfWhJZoAAAAJ&hl=zh-CN&oi=ao)<sup>1</sup>

<sup>1</sup>Beijing Institute for General Artificial Intelligence,
<sup>2</sup>ShanghaiTech University,
<sup>3</sup>Beihang University

# R-Tac Sensor Guide
### Low-cost, 120fps, Round-shape Single-channel Vision-based Tactile Sensor
[[Project page]](https://peilin-666.github.io/projects/PP-Tac/)
[[Paper]](https://arxiv.org/abs/2504.16649)
[[Video Tutorial_Youtube]](https://www.youtube.com/watch?v=-npk0cRKWGI)
[[Video Tutorial_Bilibili]](https://www.bilibili.com/video/BV1pLKwzkEMb/?spm_id_from=333.337.search-card.all.click)

<img width="90%" src="source/guide.png">

## CAD Models</a>
All CAD models can be downloaded and edited from [[onshape]](https://cad.onshape.com/documents/9472112f26da7f3a688b4739/w/3c5ca2de4889759493b1cc6c/e/c49e7e8ea4181ab75b1e7674)
(right click and copy link address to enter).<br>
3D printed files (print with supportings) can be found in [3Dprinting](Model/3Dprinting).

## Camera</a>
We use 120fps OV9281 Monochrome camera module which can be bought from [Taobao](https://item.taobao.com/item.htm?_u=u1tj68hd552c&id=740654464379&pisk=gZna_DMwEhKwG0VTSDZVTIknu9qTXoRWimNbnxD0C5VMkRtc0xkuhddvB2ymGXpvBlM0gjlQ9iwXWIIqTfMo5PGj5S20tXD1fStTujkSwjsXXVwc0XMYojiqHiymnjpTGdLIBAEYmQO7g3MtBJlqX5nai6Xnp-BGmeaG7Hk6dQOWVn8GKoRwNj_MG-83U8q0IP2gxpybn-2mir23tWeYn12cjpJU9-ygnS4iK62zhOqcjSDh-8wlmZV0s6x39-q0ijqDLkVLnfPgmuIUUPvxSnj1hIZY77DgTgk-DR01WANF3tmE8PPlVWScmDyZp7uIZg8bZ4F_P5hHAGrqK-lUPVRPbjkmeq4E01vsZxctYlEyldUr-2g772RG0Wis8DrsjIQ-dvPgzkgBJCabbfu8wPYp9jkU1jma-1RQG4MIjylkOMlYuqm_xcAH4g8dMJVRBm3NiZzgpJPWLpWVhqJbp0L0MZQYW9eUNdpOkZUgpJPWLp7AkPdLL799B&spm=a1z09.2.0.0.7c6f2e8dTlGL9i) or [Amazon](https://www.amazon.co.uk/Global-Shutter-Camera-Detection-Modules-100-degrees/dp/B0D5SNTHSD/ref=sr_1_11?crid=2R23XLE5BSYKH&dib=eyJ2IjoiMSJ9.gA3Y_wyUXfWGtyqmUP0W8uhaOtlGlJwBw8w8nT7Cm9Tn-N7f2UTGKzxSLvTmrH0WFhyYXvINhPoG-X2aafG56-EZ_NEBnZzTOUSfDsTdBUi7AcLWAB7GRZKhZwq8c-t_ErVAtIW3Mwx3082KfpEOhQ.sxC83F-wssS75YhGELiLQ7isbcCWWrhWGU-3zvZtcMA&dib_tag=se&keywords=Ov9281&qid=1748600054&sprefix=ov9281%2Caps%2C296&sr=8-11)

## PCBs</a>
The PCB files can be downloaded from [PCB](PCB/) and is powered by 5V.<br>
If you opt for white LED chips, it is recommended to select the 2835 package type and pay attention to the anode and cathode of the LED chips.<br>
<div style="display: flex; align-items: center;">
  <img src="PCB/PCB1.png" alt="Image 1" style="height: 200px;">
  <img src="PCB/PCB.png" alt="Image 2" style="height: 200px;">
</div>

## Diffuser Boards</a>
Diffuser patterns (.dxf) can be downloaded from [Diffuser](Model/Diffuser). It is recommended to laser-cut the [double-sided frosted one](https://detail.tmall.com/item.htm?_u=u1tj68hd2040&id=550079594688&pisk=gz4QsiAykpvCu90JNDfw5a6vTQ07V17VyQG8i7Lew23KVUFUhb8r4McWPANAtvSlx8Z_eRmEpYorPbwqIW8PtW55P7Fxp4rELbeaFRxULzkzQaN0hD8UwzR3j-Pv8ySnzUgnr4BV3Z7qTW0oyX7s6w9n6jGRzUnJv2093vOFDZ74t5OE9sPOuz79ZilB2Ue-vVLtaAhKwULd1chrZ4LK2eK9BADtypKK9CFtZjdp98hpW5hoiBKpvYK96XDte438e5CsaAH-9vQs_MMLO6J8xoTCqBPSCUL85fBnHW69zfaZOqkbAy_Pz_cIAxFKCtW0K7nTZ0akZUia9o2oMJpdFY4_6yEYHO-EpoG7guwAaHcZVy0seyQDGAUQNPi3bZpsslFU2y2BNUh38roSnc1Jr0uTumEYdQf062h8Em4PneDLwS400q_BnXZTGVsrYEkfnW-W1mYS1x5113xoucPAK1xN7qnKsfJV119BqDhi1x5113xovfcNg116d3f..&spm=a1z09.2.0.0.137e2e8dJizF4s). You can also try 3D printing with white resin. <br>

## Metallic Mold</a>
The design files can be downloaded from [Mold](Model/MetalMold). Two molds are needed for PDMS casting (27mm diameter) and Silicone casting (30mm diameter). The mold is fabricated using stainless steel with polished and buffing cavity.<br>

<img width="100%" src="Model/MetalMold/MetalMolds.png">

## BOM</a>

| Component | Description | Remark |
|------|------|------|
| PDMS | Transparent base | [SYLGARD™ 184](https://www.dow.com/en-us/pdp.sylgard-184-silicone-elastomer-kit.01064291z.html#overview) |
| Silicone | Translucent layer | [Ecoflex™ 00-10](https://www.smooth-on.com/products/ecoflex-00-10/) |
| Black pigment | Coating colorant | [Silc Pig™](https://www.smooth-on.com/products/silc-pig/) |
| Coating | Coating layer | [Psycho Paint™](https://www.smooth-on.com/products/psycho-paint/) |
| Thinner | Coating layer thinner | [NOVOCS™ Matte](https://www.smooth-on.com/products/novocs/) |
| Release agent | Silicone release agent | [Ease Release™ 200](https://www.smooth-on.com/products/ease-release-200/) |
| Bolts | M2 * 4 & M1.6 * 5 & M3 *4 | bolts |
| Hot melt nuts | M2 * 2 * 3 | nuts |
| Soldering iron | Compatiable with above nuts | [Soldering Iron](https://item.taobao.com/item.htm?_u=i2jo70qs4f40&id=676644094694&spm=a1z09.2.0.0.32402e8dfa9bP4)|
| Scale | For silicone mixing | Electronic Scale |
| Tape | Double-sided tape | Tape |
| Stick | Stirring silicone | Small Wood Stick |
| Cup | Container for silicone | Disposable Cups |
| Sticker | Dust removal of silicone surface | [Sticker](https://item.taobao.com/item.htm?_u=i2jo70qsf114&id=561103645534&spm=a1z09.2.0.0.32402e8dfa9bP4)|
| Vacuum pump | Air degassing for silicone | [Vacuum Pump](https://detail.tmall.com/item.htm?_u=i2jo70qs227b&id=593806957297&spm=a1z09.2.0.0.32402e8dfa9bP4&skuId=4811795832968)|
| Other tools | Small scissor, tweezer, knife | Tools |

## Video Guide</a>
[[YouTube]](https://www.youtube.com/watch?v=-npk0cRKWGI)
[[Bilibili]](https://www.bilibili.com/video/BV1pLKwzkEMb/?spm_id_from=333.337.search-card.all.click)

# Sensor Calibration
**Recommended camera resolution: 640 * 480**
## 0. Camera Intrinsic Calibration</a>
Refer to [this website](https://github.com/dyfcalid/CameraCalibration/tree/master/IntrinsicCalibration) for intrinsic calibration.
## 1. Camera Extrinsic Calibration</a>
3D print the [calibration tools](Calibration/3Dprint): top, surrounding and 5 pins. Drill and clean the holes in the top model to ensure that pins can be inserted.<br>

In accordance with the hole index in the figure below and referring to the **capture name** in this [folder](Calibration/sensor1/calib_extrinsic), insert the five pins into the corresponding holes in one go. Subsequently, save and replace the existing images.
<div style="display: flex; align-items: center;">
  <img src="source/cali1.png" alt="Image 1" style="height: 300px;">
  <img src="source/cali2.png" alt="Image 2" style="height: 260px;">
</div>

run ./Calibration/1CameraExtrinsics.py and replace the **CAMERA_MATRIX** and **Distortion** with your own camera intrinsc parameters to obtain the camera extrinsics.

## 2. Obtain the Reference Depth Map</a>
run ./Calibration/2Reprojection.py to obtain the reference depth map and the mask center and radius.

## 3. Surface Reconstruction</a>
you can either directly run ./Calibration/3ShapeReconstruction_directuse.py to obtain the surface shape via the model we have trained using indenters 
<br>
or 
<br>
you can build a table by yourself using a single ball capture (press on the top middle area), by running ./Calibration/3SensorCalibration.py and ./Calibration/4ShapeReconstruction.py

## Acknowledgment</a>
[9DTact: A Compact Vision-Based Tactile Sensor for Accurate 3D Shape Reconstruction and Generalizable 6D Force Estimation](https://arxiv.org/abs/2308.14277), Lin et al., RAL 2023

## Reference
```
@article{lin2025pp,
  title={PP-Tac: Paper Picking Using Tactile Feedback in Dexterous Robotic Hands},
  author={Lin, Pei and Huang, Yuzhe and Li, Wanlin and Ma, Jianpeng and Xiao, Chenxi and Jiao, Ziyuan},
  journal={arXiv preprint arXiv:2504.16649},
  year={2025}
}
```

<!-- 
```
@INPROCEEDINGS{lin2025pp, 
    author={Lin, Pei and Huang, Yuzhe and Li, Wanlin and Ma, Jianpeng and Xiao, Chenxi and Jiao, Ziyuan},
    title     = {PP-Tac: Paper Picking Using Tactile Feedback in Dexterous Robotic Hands},
    booktitle = {Robotics Science and Systems (RSS)}, 
    year      = {2025}, 
} 
```
-->

