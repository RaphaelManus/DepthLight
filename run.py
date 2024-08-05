# Description: Create an emissive mesh representation of a scene given an image input

import argparse
import os
import subprocess

parser = argparse.ArgumentParser(prog="python run.py",
                                    description="Create an emissive mesh representation of a scene given an image input")
parser.add_argument('-i', '--input_path', help='path of the input folder', required=True)
parser.add_argument('--type', '-t', default="ldr_lfov", choices=["ldr_lfov", "l", "ldr_pano", "p", "hdr_pano", "h"])
parser.add_argument('-o', '--output_path', help='path of the output folder, default is next to the input folder')
parser.add_argument('-f', '--fov', default="90", help='field of view of the input image, default is 90')
parser.add_argument('-p', '--optional_prompt', '--prompt', default="", help='OPTIONAL: prompt for the panorama generation, example: "indoor" or "outdoor"')
parser.add_argument('--ue', action='store_true', help='OPTIONAL: create a usd file for Unreal Engine')

args = parser.parse_args()

######################################################### CHECK CHECKPOINTS #########################################################

if not os.path.exists("./scripts/panodiff/norota_clean.ckpt"):
    cmd = "wget https://huggingface.co/gqy2468/PanoDiff/resolve/main/pretrained_models/norota_clean.ckpt -O norota_clean.ckpt"
    process = subprocess.run(cmd, shell=True, cwd="./scripts/panodiff/")

if not os.path.exists("./scripts/panoLANet/checkpoint_panoLANet/panoLANet.meta"):
    cmd = "wget https://huggingface.co/RaphaelManus/PanoLANet/resolve/main/panoLANet.data-00000-of-00001 -O panoLANet.data-00000-of-00001 && wget https://huggingface.co/RaphaelManus/PanoLANet/resolve/main/panoLANet.index -O panoLANet.index && wget https://huggingface.co/RaphaelManus/PanoLANet/resolve/main/panoLANet.meta -O panoLANet.meta"
    process = subprocess.run(cmd, shell=True, cwd="./scripts/panoLANet/checkpoint_panoLANet")

if not os.path.exists("./scripts/depth_anything_v2/checkpoints/depth_anything_v2_vitl.pth"):
    os.makedirs("./scripts/depth_anything_v2/checkpoints", exist_ok=True)
    cmd = "wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true -O depth_anything_v2_vitl.pth"
    process = subprocess.run(cmd, shell=True, cwd="./scripts/depth_anything_v2/checkpoints")

############################################################### END CHECKPOINTS ########################################################

################################################################### IMPORTS ############################################################

import cv2
import numpy as np
import torch
import imghdr
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

from scripts.depth import depthany
from scripts.diffusion import get_model, panodiff, log_images

from pxr import Usd, UsdGeom, Gf, Vt, Sdf, UsdShade

################################################################# END IMPORTS ##########################################################

#################################################################### ARGS ##############################################################

####### basic args #######
panodiff_model_path = './scripts/panodiff/models/norota_inpaint.yaml'
theta = 0
phi = 0 # change if TILT is needed

####### TODO deal with any input size #######
width = 1024
height = 512

input_path = args.input_path
fov = int(args.fov)
prompt = args.optional_prompt
if args.output_path:
    output_path = args.output_path
    ldrpano_path = os.path.join(output_path, "ldr_pano")
    hdrpano_path = os.path.join(output_path, "hdr_pano")
    usd_path = os.path.join(output_path, "usd")
else:
    output_path = os.path.dirname(input_path)
    ldrpano_path = os.path.join(output_path, "ldr_pano")
    hdrpano_path = os.path.join(output_path, "hdr_pano")
    usd_path = os.path.join(output_path, "usd")
HdrMap = "../hdr_pano" # relative path to the hdr from the usd file

############################################################### END ARGS ################################################################

############################################################## FUNCTIONS ################################################################

class Perspective:
    # equirectangular projection
    def __init__(self, img_name , FOV, THETA, PHI ):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
        self.wFOV = FOV
        self.THETA = THETA
        self.PHI = PHI
        #self.hFOV = float(self._height) / self._width * FOV
        self.hFOV = 2 * np.degrees(np.arctan(np.tan(np.radians(FOV / 2)) * (float(self._height) / self._width)))

        self.w_len = np.tan(np.radians(self.wFOV / 2.0))
        self.h_len = np.tan(np.radians(self.hFOV / 2.0))

    

    def GetEquirec(self,width,height):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        x,y = np.meshgrid(np.linspace(-180, 180,width),np.linspace(90,-90,height))
        
        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map,y_map,z_map),axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(self.THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-self.PHI))

        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R2, xyz)
        xyz = np.dot(R1, xyz).T

        xyz = xyz.reshape([height , width, 3])
        inverse_mask = np.where(xyz[:,:,0]>0,1,0)

        xyz[:,:] = xyz[:,:]/np.repeat(xyz[:,:,0][:, :, np.newaxis], 3, axis=2)
        
        
        lon_map = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),(xyz[:,:,1]+self.w_len)/2/self.w_len*self._width,0)
        lat_map = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),(-xyz[:,:,2]+self.h_len)/2/self.h_len*self._height,0)
        mask = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),1,0)

        persp = cv2.remap(self._img, lon_map.astype(np.float32), lat_map.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        
        mask = mask * inverse_mask
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        persp = persp * mask
        
        
        return persp , mask
    
class EmissiveMesh:
    # depth inference and mesh creation
    def SphereGrid(equ_h, equ_w):
        cen_x = (equ_w - 1) / 2.0
        cen_y = (equ_h - 1) / 2.0
        theta = (2 * (np.arange(equ_w) - cen_x) / equ_w) * np.pi
        phi = (2 * (np.arange(equ_h) - cen_y) / equ_h) * (np.pi / 2)
        theta = np.tile(theta[None, :], [equ_h, 1])
        phi = np.tile(phi[None, :], [equ_w, 1]).T

        x = (np.cos(phi) * np.sin(theta)).reshape([equ_h, equ_w, 1])
        y = (np.sin(phi)).reshape([equ_h, equ_w, 1])
        z = (np.cos(phi) * np.cos(theta)).reshape([equ_h, equ_w, 1])
        xyz = np.concatenate([x, y, z], axis=-1)

        return xyz
    
    def create_mesh(ldrpano):
        if imghdr.what(ldrpano) == "exr":
            raw_image = cv2.imread(ldrpano, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        else:
            raw_image = cv2.imread(ldrpano, cv2.COLOR_BGR2RGB)
        depth = depthany(raw_image)
        
        depth = 1/(depth+10e-10) # deal with inf distance values - TO CHANGE -
        range_max = depth.min()*np.minimum(depth.max() / (depth.min() + 10e-10), 100.0)
        depth = (depth) / (range_max - depth.min())
        depth = np.power(depth, 1.0 / 2.2)
        depth = depth * 10

        grid = EmissiveMesh.SphereGrid(*depth.shape)
        pts = depth[..., None] * grid
        pts = pts.reshape([-1, 3])

        # -------------------------------- Create the faces -------------------------------- #

        faces = np.zeros([(pts.shape[0]-width)*2, 3], dtype=np.int32)

        for i in range(height-1):
            for j in range(width-1):
                faces[(i*width+j)*2] = [i*width+j, (i+1)*width+j, i*width+j+1]
                faces[(i*width+j)*2+1] = [i*width+j+1, (i+1)*width+j, (i+1)*width+j+1]
            # case for last column
            faces[(i*width+width-1)*2] = [i*width+width-1, (i+1)*width+width-1, i*width]
            faces[(i*width+width-1)*2+1] = [i*width, (i+1)*width+width-1, (i+1)*width]

        # --------------- duplicate everything to give thickness to the mesh --------------- #

        if args.ue:
            # apply a 10% scale to the second mesh
            pts2 = pts.copy()
            pts2[:, 0] *= 1.1
            pts2[:, 1] *= 1.1
            pts2[:, 2] *= 1.1

            # Recreate the faces for the second mesh with opposite winding order
            faces2 = np.zeros([(pts.shape[0]-width)*2, 3], dtype=np.int32)

            for i in range(height-1):
                for j in range(width-1):
                    faces2[(i*width+j)*2] = [i*width+j, i*width+j+1, (i+1)*width+j]
                    faces2[(i*width+j)*2+1] = [i*width+j+1, (i+1)*width+j+1, (i+1)*width+j]
                # case for last column
                faces2[(i*width+width-1)*2] = [i*width+width-1, i*width, (i+1)*width+width-1]
                faces2[(i*width+width-1)*2+1] = [i*width, (i+1)*width, (i+1)*width+width-1]

            faces2 += pts.shape[0]

            # merge the two meshes
            pts = np.concatenate([pts, pts2], axis=0)
            faces = np.concatenate([faces, faces2], axis=0)

        # switch y and z axis
        pts[:, [1, 2]] = pts[:, [2, 1]]

        return pts, faces

    def create_usd_file(pts, faces, HdrMap, UsdFile):
        # ------------------------------ Create the USD file ------------------------------ #
        # check if the file already exists
        if os.path.exists(UsdFile):
            print(f"File {UsdFile} will be overwritten")
            # delete the file if it already exists
            os.remove(UsdFile)
        stage = Usd.Stage.CreateNew(UsdFile) # replace with CreateInMemory()

        # ------------------------------ Scene properties --------------------------------- #
        meterperunit = UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        # set down axis to -Y
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        # ------------------------ Create a mesh from our arrays -------------------------- #

        # Create a mesh
        #start with a xform
        xform = UsdGeom.Xform.Define(stage, '/myMesh')
        mesh = UsdGeom.Mesh.Define(stage, '/myMesh/mesh')

        # Set points
        points = mesh.CreatePointsAttr()
        # swap y and z
        points.Set(-pts)

        # Set faces
        faceVertexCounts = mesh.CreateFaceVertexCountsAttr()
        faceVertexCounts.Set([3]*len(faces))

        faceVertexIndices = mesh.CreateFaceVertexIndicesAttr()
        faceVertexIndices.Set(faces.flatten())

        # ------------------------------ Texture coordinates ------------------------------ #

        texCoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('UVMap', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        # go through each face and assign the texture coordinates as i/(width-1), j/(height-1), we only need one set of texture coordinates for each vertex
        tex = []
        for i in range(height):
            for j in range(width):
                tex.append([j/(width-1), 1.0-i/(height-1)])
        if args.ue:
            # duplicate the texture coordinates for the second layer
            for i in range(height):
                for j in range(width):
                    tex.append([j/(width-1), 1.0-i/(height-1)])
        texCoords.Set(Vt.Vec2fArray(tex))

        # ------------------------------- Create a material ------------------------------ #

        # store material in _materials
        _materials = stage.DefinePrim('/_materials')
        material = UsdShade.Material.Define(stage, '/_materials/Material')
        shader = UsdShade.Shader.Define(stage, '/_materials/Material/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')

        # Set the shader inputs
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))  # White color
        shader.CreateInput('emissiveIntensity', Sdf.ValueTypeNames.Float).Set(10.0)  # Intensity

        # Connect the shader to the material
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')

        # read the texture file
        stReader = UsdShade.Shader.Define(stage, '/_materials/Material/stReader')
        stReader.CreateIdAttr('UsdPrimvarReader_float2')

        diffuseTexture = UsdShade.Shader.Define(stage, '/_materials/Material/diffuseTexture')
        diffuseTexture.CreateIdAttr('UsdUVTexture')
        # if HdrMap starts with ./ remove it
        if HdrMap.startswith('./'):
            HdrMap = HdrMap[2:]
        diffuseTexture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(HdrMap)
        diffuseTexture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
        diffuseTexture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTexture.ConnectableAPI(), 'rgb')

        stReader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('UVMap')

        # Assign the material to the mesh
        mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

        # ------------------------------ Save the USD file ------------------------------ #
        stage.GetRootLayer().Save()

############################################################## END FUNCTIONS ################################################################

################################################################## MAIN #####################################################################

if __name__ == '__main__':

    if args.type == "ldr_lfov" or args.type == "l" :
        panodiff_model = get_model()
        panodiff_model = panodiff_model.cuda()

    for i, img in enumerate(os.listdir(input_path)):

        img_name = img.split(".")[0]
        img_ext = img.split(".")[1]

        if args.type == "ldr_lfov" or args.type == "l" :
            per = Perspective(os.path.join(input_path, img), fov, theta, phi)
            equi, mask = per.GetEquirec(1024, 512)
            # turn masked area to alpha channel
            equi = np.concatenate([equi, np.ones_like(equi[:, :, 0:1]) * 255], axis=2)
            equi[:, :, 3] = mask[:, :, 0] * 255
            target = equi[:, :, :3]
            mask = mask[:, :, 0]
            mask = np.where(mask>0, 1.0, 0.0).astype(np.float32)
            target = target.astype(np.float32)

            dataloader = panodiff(target, mask, prompt)

            with torch.no_grad():
                batch = next(iter(dataloader))

                for item in batch:
                    if isinstance(batch[item], torch.Tensor):
                        batch[item] = batch[item].to(panodiff_model.device)

                kwargs = {}
                #images = panodiff_model.log_images(batch, split="test", **kwargs)
                images = log_images(panodiff_model, batch, **kwargs)
                os.makedirs(ldrpano_path, exist_ok=True)

                image = images['samples_cfg_scale_9.00'][0]
                image = torch.clamp(image.detach().cpu(), -1., 1.)
                image = (image + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                image = image.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
                image = (image * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(ldrpano_path, img_name + "." + img_ext), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                torch.cuda.empty_cache()

        if args.type == "ldr_pano" or args.type == "p" :
            ldrpano_path = input_path
        if not (args.type == "hdr_pano" or args.type == "h") :
            cmd = f"conda run -n LANet --live-stream python scripts/panoLANet/src/main.py --phase test --gpu 0 --checkpoint_dir ./scripts/panoLANet/checkpoint_panoLANet/ --test_dir " + ldrpano_path + " --out_dir " + hdrpano_path
            print(cmd)
            process = subprocess.run(cmd, shell=True, cwd="./")

        if args.type == "hdr_pano" or args.type == "h" :
            ldrpano_path = input_path # input is hdr, so we don't need to create the ldrpano
            hdrpano_path = input_path # input is hdr, so we don't need to create the hdrpano

            # TODO deal with case when input path is not the proper relative path to the usd file
        pts, faces = EmissiveMesh.create_mesh(os.path.join(ldrpano_path, img_name + "." + img_ext))
        EmissiveMesh.create_usd_file(pts, faces, os.path.join(HdrMap, img_name + ".exr"), os.path.join(usd_path, img_name + ".usda")) # TODO deal with png or exr inputs

        print(f"Created DepthLight mesh for {img}")
