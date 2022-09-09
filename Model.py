import numpy as np
import pandas as pd
import open3d as o3d
import ntpath
from alphashape import alphashape

dr = np.pi/180
class Geometry3D:
    def __init__(self, filepath = None, name = None, scale = 1):
        if filepath == None and name is not None:
            self.geometry_type = o3d.io.CONTAINS_POINTS
            self.name = name
            self.cloud = o3d.geometry.PointCloud()
        else:
            self.name = ntpath.basename(filepath)
            self.geometry_type = o3d.io.read_file_geometry_type(filepath)
            self.scale = scale
            if self.geometry_type & o3d.io.CONTAINS_TRIANGLES:
                self.mesh = o3d.io.read_triangle_mesh(filepath)
                self.mesh = self.mesh.scale(scale,self.mesh.get_center())
                self.wire = o3d.geometry.LineSet.create_from_triangle_mesh(self.mesh)
                self.mesh.compute_vertex_normals()
                self.mesh.paint_uniform_color([1, 1, 1])
                self.wire.paint_uniform_color([0, 0, 0])
                print('[Info] Successfully read', filepath)
            elif self.geometry_type & o3d.io.CONTAINS_POINTS:
                self.z_direction = 0
                self.cloud = o3d.io.read_point_cloud(filepath)
                self.cloud = self.cloud.scale(scale,self.cloud.get_center())
                self.cloud.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 1, max_nn = 25))
                self.cloud.estimate_covariances(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 1, max_nn = 25))
                self.cloud.normalize_normals()
                self.cloud.paint_uniform_color([0,0,1])
                print('[Info] Successfully read', filepath)
            else:
                print('[WARNING] Failed to read file:', filepath)

class Surface_XY:
    def __init__(self, filepath):
        self.Dataframe = pd.read_excel(filepath,sheet_name='Active parameters')
        self.name = ntpath.basename(filepath)
        self.sur = self.Dataframe.columns[1:]

    def Sag_Z(self,X,Y,S):
        C = self.Dataframe[S].values
        #X *= -1
        R2 = X**2 + Y**2
        Z = C[6]*(R2)/(1+np.sqrt(1-(1+C[7])*C[6]**2*(R2)))+C[8]
        n,i,j = 1,1,0
        for c in C[9:]:
            if c == 0:
                pass
            else:
                Z += c*Y**i*X**j
            if n==j:
                n,i,j = n+1,n+1,0
            else:
                i,j = i-1, j+1
        return Z

    def Matrix44(self,S):
        C = self.Dataframe[S].values
        alpha,beta,gamma,x0,y0,z0 = C[:6]
        alpha,beta,gamma = alpha*dr,beta*dr,gamma*dr
        Shift = np.array([[1,0,0,x0],[0,1,0,y0],[0,0,1,z0],[0,0,0,1]])
        Rx = np.array([[1,0,0,0],[0,np.cos(alpha),np.sin(alpha),0],[0,-np.sin(alpha),np.cos(alpha),0],[0,0,0,1]])
        Ry = np.array([[np.cos(beta),0,-np.sin(beta),0],[0,1,0,0],[np.sin(beta),0,np.cos(beta),0],[0,0,0,1]])
        Rz = np.array([[np.cos(gamma),-np.sin(gamma),0,0],[np.sin(gamma),np.cos(gamma),0,0],[0,0,1,0],[0,0,0,1]])
        return Shift.dot(Rx.dot(Ry.dot(Rz)))

    def Surface_edge(self,pcd,S):
        pcd.transform(np.linalg.inv(self.Matrix44(S)))
        xyz = np.asarray(pcd.points)
        zt = self.Sag_Z(*xyz[:,:2].T,S)
        index = np.where(np.abs(zt-xyz[:,2])<0.001)[0]
        if len(index)>0:
            edge = np.asarray(alphashape(xyz[index,:2],0.2).exterior.coords)
            idx = [np.argmin(np.sum(np.square(xyz[:,:2]-i),axis = 1)) for i in edge]
            vol = o3d.visualization.SelectionPolygonVolume()
            vol.bounding_polygon = o3d.utility.Vector3dVector(1.001*xyz[idx])
            vol.axis_max, vol.axis_min= np.inf,-np.inf
            vol.orthogonal_axis='Z'
        else:
            vol = None
        return vol,index

    def Generate_points(self,pcd,S,Xs,Ys):
        Edge,inside_idx = self.Surface_edge(pcd,S)
        pcd = pcd.select_by_index(inside_idx)
        bounding_box = pcd.get_axis_aligned_bounding_box()
        xmin,ymin,_ = bounding_box.get_min_bound()
        xmax,ymax,_ = bounding_box.get_max_bound()
        x,y = np.arange(xmin,xmax+Xs,Xs),np.arange(ymin,ymax+Ys,Ys)
        x,y = np.meshgrid(x, y)
        x,y = x.reshape((-1)),y.reshape((-1))
        z = self.Sag_Z(x,y,S)
        pcd.points = o3d.utility.Vector3dVector(np.vstack((x,y,z)).T)
        pcd = Edge.crop_point_cloud(pcd)
        pcd.transform(self.Matrix44(S))
        pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 1, max_nn = 25))
        pcd.estimate_covariances(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 1, max_nn = 25))
        pcd.normalize_normals()
        return pcd
        