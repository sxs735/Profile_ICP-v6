from copy import deepcopy
import numpy as np
import pandas as pd
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from matplotlib.colors import ColorConverter, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import Model

Weight_Function = {'None': None,
                   'CauchyLoss' : o3d.pipelines.registration.CauchyLoss(k=0.05),
                   'GMLoss': o3d.pipelines.registration.GMLoss(k=0.05),
                   'TukeyLoss': o3d.pipelines.registration.TukeyLoss(k=0.05),
                   'HuberLoss': o3d.pipelines.registration.HuberLoss(k=0.05),
                   'L1Loss': o3d.pipelines.registration.L1Loss(),
                   'L2Loss': o3d.pipelines.registration.L2Loss()}
ICP_Class = {'PointToPoint' : o3d.pipelines.registration.TransformationEstimationPointToPoint(),
             'PointToPlane' : o3d.pipelines.registration.TransformationEstimationPointToPlane(),
             'Generalized' : o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()}

cmaps_dir = {'Uniform Sequential':['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
             'Sequential': ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'],
             'Sequential (2)': ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink','spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia','hot', 'afmhot', 'gist_heat', 'copper'],
             'Diverging': ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu','RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'],
             'Cyclic': ['twilight', 'twilight_shifted', 'hsv'],
             'Qualitative': ['Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2', 'Set1', 'Set2', 'Set3','tab10', 'tab20', 'tab20b', 'tab20c'],
             'Miscellaneous': ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern','gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg','gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral','gist_ncar']}

class AppWindow:
    Model_lib = {}
    Data_list = []
    Target_list = []
    Selected = {}

    #Default_value
    State = 'Main'
    File_Load = [None,1]
    Sampling = [0, 0.5, 0.5]
    ColorMap = ['Miscellaneous','gist_rainbow','_r',False,np.nan,np.nan]
    ICP_parameter = [True,'PointToPoint',0.1,'None',np.nan]

    # Material
    material_cloud = rendering.MaterialRecord()
    material_cloud.point_size = 12
    material_cloud.shader = 'defaultUnlit'
    material_wire = rendering.MaterialRecord()
    material_wire.base_color = [0, 0, 0, 0]
    material_wire.shader = 'unlitLine'
    material_wire.line_width = 2
    material_mesh = rendering.MaterialRecord()
    material_mesh.base_color = [0.7, 0.7, 0.7, 1]
    material_mesh.shader = 'defaultUnlit'
    material_sphere = rendering.MaterialRecord()
    material_sphere.shader = 'defaultUnlit'

    def __init__(self, width, height):
        self.app = gui.Application.instance
        self.window = self.app.create_window(self.State, width, height)
        self.em = self.window.theme.font_size
        separation_height = int(round(0.5 * self.em))

        self.Manual = gui.Label('Manual')
        self.Manual.background_color = gui.Color(0,0,0,0)
        self.Manual.text_color = gui.Color(0,0,0,1)
        self.Manual.visible = True

        #Target
        self.Target_button_load = gui.Button('Load')
        self.Target_button_load.horizontal_padding_em = 0
        self.Target_button_load.vertical_padding_em = 0
        self.Target_button_load.set_on_clicked(self.Target_Load_clicked)

        self.Target_button_del = gui.Button('Del')
        self.Target_button_del.horizontal_padding_em = 0
        self.Target_button_del.vertical_padding_em = 0
        self.Target_button_del.set_on_clicked(self.Target_Delete)
        self.Target_button_del.enabled = False

        self.Target_View = gui.ListView()
        self.Target_View.set_max_visible_items(6)
        self.Target_View.set_on_selection_changed(self.Target_View_mouse)

        self.Change_Tbutton = gui.Button('')
        self.Change_Tbutton.horizontal_padding_em = 0.5
        self.Change_Tbutton.vertical_padding_em = 0
        self.Change_Tbutton.set_on_clicked(self.To_Data)
        self.Change_Tbutton.enabled = False

        target_bar = gui.Horiz(0.25 * self.em)
        target_bar.add_child(gui.Label('Target'))
        target_bar.add_child(self.Change_Tbutton)
        target_bar.add_stretch()
        target_bar.add_child(self.Target_button_load)
        target_bar.add_child(self.Target_button_del)

        self.Coeff_name = gui.TextEdit()
        self.Coeff_name.enabled = False
        self.Coeff_button_load = gui.Button('Load')
        self.Coeff_button_load.horizontal_padding_em = 0
        self.Coeff_button_load.vertical_padding_em = 0
        self.Coeff_button_load.set_on_clicked(self.Coeff_Load_dialog)

        Coeff_bar = gui.Horiz(0.25 * self.em)
        Coeff_bar.add_child(self.Coeff_name)
        Coeff_bar.add_child(self.Coeff_button_load)
        
        #Data
        self.Data_button_load = gui.Button('Load')
        self.Data_button_load.horizontal_padding_em = 0
        self.Data_button_load.vertical_padding_em = 0
        self.Data_button_load.set_on_clicked(self.Data_Load_clicked)

        self.Data_button_del = gui.Button('Del')
        self.Data_button_del.enabled = False
        self.Data_button_del.horizontal_padding_em = 0
        self.Data_button_del.vertical_padding_em = 0
        self.Data_button_del.set_on_clicked(self.Data_Delete)

        self.Data_View = gui.ListView()
        self.Data_View.set_max_visible_items(8)
        self.Data_View.set_on_selection_changed(self.Data_View_mouse)

        self.Change_Dbutton = gui.Button('')
        self.Change_Dbutton.horizontal_padding_em = 0.5
        self.Change_Dbutton.vertical_padding_em = 0
        self.Change_Dbutton.set_on_clicked(self.To_Target)
        self.Change_Dbutton.enabled = False

        self.ICP_button = gui.Button('Manual ICP')
        self.ICP_button.horizontal_padding_em = 1.5
        self.ICP_button.vertical_padding_em = 1
        self.ICP_button.set_on_clicked(self.Manual_ICP)
        self.ICP_button.enabled = False

        data_bar = gui.Horiz(0.25 * self.em)
        data_bar.add_child(gui.Label('Data'))
        data_bar.add_child(self.Change_Dbutton)
        data_bar.add_stretch()
        data_bar.add_child(self.Data_button_load)
        data_bar.add_child(self.Data_button_del)

        #Sag Error
        self.SagErr_cal_button = gui.Button('Sag Error Cal.')
        self.SagErr_cal_button.enabled = False
        self.SagErr_cal_button.horizontal_padding_em = 0
        self.SagErr_cal_button.vertical_padding_em = 0
        self.SagErr_cal_button.set_on_clicked(self.SagErr_cal)

        SagErr_bar = gui.Horiz(0.25 * self.em)
        SagErr_bar.add_stretch()
        SagErr_bar.add_child(self.SagErr_cal_button)

        self.Console = gui.Vert(0, gui.Margins(0.5*self.em, 0.5*self.em, 0.25*self.em, 0.5*self.em))
        self.Console.add_child(gui.Label('Surface parameter'))
        self.Console.add_child(Coeff_bar)
        self.Console.add_fixed(2*separation_height)
        self.Console.add_child(target_bar)
        self.Console.add_child(self.Target_View)
        self.Console.add_fixed(2*separation_height)
        self.Console.add_child(data_bar)
        self.Console.add_child(self.Data_View)
        self.Console.add_fixed(0.5*separation_height)
        self.Console.add_child(SagErr_bar)
        self.Console.add_fixed(2*separation_height)
        self.Console.add_child(self.ICP_button)

        #Detail panel
        self.Direction = gui.Combobox()
        self.Direction.add_item('+')
        self.Direction.add_item('-')
        self.Direction.set_on_selection_changed(self.Apply_enabled)
        self.Scale_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.Scale_value.enabled = False
        self.Max_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.Max_value.enabled = False
        self.Min_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.Min_value.enabled = False
        self.Avg_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.Avg_value.enabled = False
        self.Std_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.Std_value.enabled = False

        Value_Grid = gui.VGrid(4, 0.1*self.em,gui.Margins(0*self.em, 0.1*self.em, 0.1*self.em, 0.1*self.em))
        Value_Grid.add_child(gui.Label('Z'))
        Value_Grid.add_child(self.Direction)
        Value_Grid.add_child(gui.Label('Scale'))
        Value_Grid.add_child(self.Scale_value)
        Value_Grid.add_child(gui.Label('Max'))
        Value_Grid.add_child(self.Max_value)
        Value_Grid.add_child(gui.Label(' Min'))
        Value_Grid.add_child(self.Min_value)
        Value_Grid.add_child(gui.Label('Avg'))
        Value_Grid.add_child(self.Avg_value)
        Value_Grid.add_child(gui.Label(' Std'))
        Value_Grid.add_child(self.Std_value)

        self.FilterMax_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.FilterMin_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.Filter_button = gui.Button('Filter')
        self.Filter_button.enabled = False
        self.Filter_button.toggleable = True
        self.Filter_button.horizontal_padding_em = 0.5
        self.Filter_button.vertical_padding_em = 0
        self.Filter_button.set_on_clicked(self.Apply_enabled)

        self.Delete_button = gui.Button('Delete')
        self.Delete_button.horizontal_padding_em = 0
        self.Delete_button.vertical_padding_em = 0
        self.Delete_button.set_on_clicked(self.Delete_mode)

        Filter_set = gui.VGrid(3, 0.1*self.em,gui.Margins(0.5*self.em, 0.1*self.em, 0.5*self.em, 0.1*self.em))
        Filter_set.add_child(self.Delete_button)
        Filter_set.add_child(gui.Label('Less than'))
        Filter_set.add_child(self.FilterMin_value)
        Filter_set.add_child(self.Filter_button)
        Filter_set.add_child(gui.Label('Bigger than'))
        Filter_set.add_child(self.FilterMax_value)

        self.Apply_button = gui.Button('Apply')
        self.Apply_button.enabled = False
        self.Apply_button.horizontal_padding_em = 0
        self.Apply_button.vertical_padding_em = 0
        self.Apply_button.set_on_clicked(self.Apply_clicked)

        self.Back_button = gui.Button('Back')
        self.Back_button.enabled = True
        self.Back_button.horizontal_padding_em = 0
        self.Back_button.vertical_padding_em = 0
        self.Back_button.set_on_clicked(self.Back_Main)

        self.Save_button = gui.Button('Save')
        self.Save_button.enabled = True
        self.Save_button.horizontal_padding_em = 0
        self.Save_button.vertical_padding_em = 0
        self.Save_button.set_on_clicked(self.Save_clicked)

        self.visible_check = gui.Checkbox('')
        self.visible_check.set_on_checked(self.Change_visible)
        Detail_button_bar = gui.Horiz(0.25 * self.em)
        Detail_button_bar.add_child(self.visible_check)
        Detail_button_bar.add_stretch()
        Detail_button_bar.add_child(self.Apply_button)
        Detail_button_bar.add_fixed(0.5*self.em)
        Detail_button_bar.add_child(self.Back_button)
        Detail_button_bar.add_fixed(0.5*self.em)
        Detail_button_bar.add_child(self.Save_button)

        self.histogram = gui.ImageWidget(o3d.geometry.Image(np.zeros((10*self.em,16*self.em,3),dtype=np.uint8)))
        self.Data_name = gui.TextEdit()
        self.Data_name.enabled = False

        Detail_tile = gui.Horiz(0.25 * self.em)
        Detail_tile.add_child(gui.Label('Name'))
        Detail_tile.add_child(self.Data_name)

        self.Detail = gui.Vert(0, gui.Margins(0.5*self.em, 0.5*self.em, 0.1*self.em, 0.5*self.em))
        self.Detail.add_child(Detail_tile)
        self.Detail.add_fixed(0.5*self.em)
        self.Detail.add_child(Value_Grid)
        self.Detail.add_child(gui.Label('Histogram'))
        self.Detail.add_child(self.histogram)
        self.Detail.add_fixed(0.5*self.em)
        self.Detail.add_child(Filter_set)
        self.Detail.add_fixed(self.em)
        self.Detail.add_child(Detail_button_bar)
        
        self.colorbar = gui.ImageWidget()
        self.colorbar.visible = False

        self.Cancel_button = gui.Button('Cancel')
        self.Cancel_button.enabled = True
        self.Cancel_button.horizontal_padding_em = 0
        self.Cancel_button.vertical_padding_em = 0
        self.Cancel_button.set_on_clicked(self.Cancel)

        self.Correct_button = gui.Button('Correct')
        self.Correct_button.enabled = True
        self.Correct_button.horizontal_padding_em = 0
        self.Correct_button.vertical_padding_em = 0

        self.Sub_panel = gui.Horiz(0, gui.Margins(0.5*self.em, 0.5*self.em, 0.25*self.em, 0.5*self.em))
        self.Sub_panel.add_stretch()
        self.Sub_panel.add_child(self.Cancel_button)
        self.Sub_panel.add_stretch()
        self.Sub_panel.add_child(self.Correct_button)
        self.Sub_panel.add_stretch()

        #tab
        self.tabs = gui.StackedWidget()
        self.tabs.add_child(self.Console)
        self.tabs.add_child(self.Detail)
        self.tabs.add_child(self.Sub_panel)
        self.Panel = [self.Console,self.Detail,self.Sub_panel]

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self._scene.set_on_mouse(self.on_mouse_widget3d)
        self._scene.set_on_key(self.on_key_widget3d)
        self.window.add_child(self._scene)

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.tabs)
        self.window.add_child(self.Manual)
        self.window.add_child(self.colorbar)

        self._RectSelect = gui.Label('')
        self._RectSelect.visible = False
        self._RectSelect.background_color = gui.Color(0,0,0,0.2)
        self.window.add_child(self._RectSelect)

        #Menu
        self.Menu = gui.Menu()
        file_menu = gui.Menu()
        file_menu.add_item("csv2xyz", 1)
        #file_menu.add_item("Export Current Image...",2)
        self.Option_menu = gui.Menu()
        self.Option_menu.add_item('Drawing settings',11)
        self.Option_menu.add_item('ICP settings',12)
        self.Option_menu.add_separator()
        self.Option_menu.add_item('Master_visible',13)
        self.Option_menu.set_checked(13, False)
        self.window.set_on_menu_item_activated(13, self.Master_visible)
        self.Menu.add_menu("File", file_menu)
        self.Menu.add_menu("Option", self.Option_menu)
        gui.Application.instance.menubar = self.Menu
        self.window.set_on_menu_item_activated(1, self.csv2xyz_dialog)
        #self.window.set_on_menu_item_activated(2, self.Export_image)
        self.window.set_on_menu_item_activated(11, self.Draw_dialog)
        self.window.set_on_menu_item_activated(12, self.ICP_dialog)

        self.Menu.set_enabled(13,False)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        visible_panel = self.Panel[self.tabs.selected_index]
        height = min(r.height,visible_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)
        self.tabs.frame = gui.Rect(r.get_right() - width, r.y, width,height)
        
        pref = self.Manual.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self.Manual.frame = gui.Rect(r.get_right()-pref.width, r.get_bottom()-pref.height, pref.width, pref.height)
        pref = self.colorbar.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self.colorbar.frame = gui.Rect(r.get_right()-width-pref.width, r.y, pref.width, pref.height)

        self.window.close_dialog()

    def csv2xyz_dialog(self):
        def csv2xyz(filepath):
            self.window.close_dialog()
            df = pd.read_csv(filepath,skiprows = 23)
            df.iloc[:,2] = pd.to_numeric(df.iloc[:,2],'coerce')
            df = df.dropna().values
            df = df-np.min(df,axis = 0)
            np.savetxt(filepath[:-3]+'xyz',df)
        dlg = gui.FileDialog(gui.FileDialog.OPEN, 'Choose file to load', self.window.theme)
        dlg.add_filter('.csv','Point cloud files (.csv)')
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(csv2xyz)
        self.window.show_dialog(dlg)

    # def Export_image(self):
    #     def Export_image(path, width, height):
    #         def on_image(image):
    #             img = image
    #             quality = 9  # png
    #             if path.endswith(".jpg"):
    #                 quality = 100
    #             o3d.io.write_image(path, img, quality)
    #         self._scene.scene.scene.render_to_image(on_image)

    #     def Export_dialog_done(filename):
    #         self.window.close_dialog()
    #         frame = self._scene.frame
    #         Export_image(filename, frame.width, frame.height)

    #     dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",self.window.theme)
    #     dlg.add_filter(".png", "PNG files (.png)")
    #     dlg.set_on_cancel(self.window.close_dialog)
    #     dlg.set_on_done(Export_dialog_done)
    #     self.window.show_dialog(dlg)


    def Coeff_Load_dialog(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, 'Choose file to load', self.window.theme)
        dlg.add_filter('.xlsx','Surface parameters files (.xlsx)')
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(self.Coeff_load_done)
        self.window.show_dialog(dlg)

    def Coeff_load_done(self,filepath):
        self.Coeff = Model.Surface_XY(filepath)
        self.Coeff_name.text_value = self.Coeff.name
        self.Sampling = [0, 0.5, 0.5]
        self.window.close_dialog()

    def Coeff_Delete(self):
        self.Coeff_name.text_value = ''
        self.Coeff_button_del.enabled = False
        self.Surface.clear_items()
        del self.Coeff

    def Target_Load_clicked(self):
        self.window.show_dialog(self.Load_Dialog('Target_Button'))

    def Data_Load_clicked(self):
        self.window.show_dialog(self.Load_Dialog('Data_Button'))

    def Load_Dialog(self,Button):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, 'Choose file to load', self.window.theme)
        if Button == 'Target_Button':
            dlg.add_filter('.stl .fbx .obj .off .gltf .glb','Triangle mesh files (.stl, .fbx, .obj, .off, .gltf, .glb)')
        elif Button == 'Data_Button':
            dlg.add_filter('.xyz .xyzn .xyzrgb .ply .pcd .pts','Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, .pcd, .pts)')
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(self.Load_dialog_done)
        return dlg

    def Load_dialog_done(self,filepath):
        self.File_Load[0] = filepath
        self.window.close_dialog()
        self.window.show_dialog(self.Scale_dialog())

    def Scale_dialog(self):
        dlg = gui.Dialog('')
        self.Scale = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.Scale.int_value = 1
        dig_done = gui.Button('Done')
        dig_done.vertical_padding_em = 0
        dig_done.set_on_clicked(self.Scaling_dialog_done)
        vert = gui.Vert(0, gui.Margins(self.em, self.em, self.em, self.em))
        vert.add_child(gui.Label('Scaling ratio'))
        vert.add_child(self.Scale)
        vert.add_fixed(0.25 * self.em)
        vert.add_child(dig_done)
        dlg.add_child(vert)
        return dlg

    def Scaling_dialog_done(self):
        self.File_Load[1] = self.Scale.double_value
        self.window.close_dialog()
        self.Load(*self.File_Load)
    
    def Load(self,filepath,scale):
        Geometry = Model.Object3D(filepath,scale = scale)
        Geometry.visible = True
        if Geometry.type & o3d.io.CONTAINS_TRIANGLES:
            for name in self.Target_list:
                self._scene.scene.remove_geometry(name)
                self._scene.scene.remove_geometry('wire_'+name)
                del self.Model_lib[name]
            self.Target_list = []
            self.Model_lib[Geometry.name] = Geometry
            self._scene.scene.add_geometry(Geometry.name, Geometry.mesh, self.material_mesh)
            self._scene.scene.add_geometry('wire_'+Geometry.name, Geometry.wire, self.material_wire)

            bounds = Geometry.mesh.get_axis_aligned_bounding_box()
            self.Target_list.append(Geometry.name)
            self.Target_View.set_items(self.Target_list)
            self.Master_name = Geometry.name
            self.Option_menu.set_checked(13, True)
            self.Menu.set_enabled(13,True)

        elif Geometry.type & o3d.io.CONTAINS_POINTS:
            Geometry.z_direction = 0
            self.Model_lib[Geometry.name] = Geometry
            self._scene.scene.add_geometry(Geometry.name, Geometry.cloud, self.material_cloud)
            bounds = Geometry.cloud.get_axis_aligned_bounding_box()
            if Geometry.name not in self.Data_list:
                self.Data_list.append(Geometry.name)
                self.Data_View.set_items(self.Data_list)
        else:
            bounds = o3d.geometry.AxisAlignedBoundingBox(np.array([-1,-1,-1]),np.array([1,1,1]))
        self._scene.setup_camera(60, bounds, bounds.get_center())

    def Target_Delete(self):
        del self.Model_lib[self.Target_View.selected_value], self.Selected['Target']
        self._scene.scene.remove_geometry(self.Target_View.selected_value)
        self._scene.scene.remove_geometry('wire_'+self.Target_View.selected_value)
        self.Target_list.remove(self.Target_View.selected_value)
        self.Target_View.set_items(self.Target_list)
        self.Change_Tbutton.enabled = False
        self.Target_button_del.enabled = False
        self.ICP_button.enabled = False

    def Data_Delete(self):
        del self.Model_lib[self.Data_View.selected_value], self.Selected['Data']
        self._scene.scene.remove_geometry(self.Data_View.selected_value)
        self.Data_list.remove(self.Data_View.selected_value)
        self.Data_View.set_items(self.Data_list)
        self.Change_Dbutton.enabled = False
        self.Data_button_del.enabled = False
        self.ICP_button.enabled = False

    def To_Target(self):
        self.Change_List('Data')
    
    def To_Data(self):
        self.Change_List('Target')

    def Change_List(self,button):
        name = self.Selected[button].name
        self.Selected[button].visible = True
        List1 = self.Target_list if button == 'Target' else self.Data_list
        List2 = self.Target_list if button == 'Data' else self.Data_list
        if name[:3] != '{T}' and '{T}'+name not in self.Model_lib:
            color = [0,0,1] if button == 'Target' else [0,0.392,0]
            self.Model_lib[name].cloud.paint_uniform_color(color)
            List1.remove(name)
            List2.append('{T}'+name)
            self.Model_lib['{T}'+name] = self.Model_lib.pop(name)
            self.Selected[button].name = '{T}'+name
            self.Target_View.set_items(self.Target_list)
            self.Data_View.set_items(self.Data_list)
            self._scene.scene.remove_geometry(name)
            self._scene.scene.add_geometry('{T}'+name,self.Selected[button].cloud,self.material_cloud)
        else:
            print(name+' do not transfer')

    def ICP_button_enabled(self):
        if 'Target' in self.Selected and self.Selected['Target'].type & o3d.io.CONTAINS_POINTS and 'Data' in self.Selected:
            return True
        else:
            return False

    def Back_Main(self):
        self.Clean_Mark()
        del self.active_model,self.label_list,self.picked_idx
        self.tabs.selected_index = 0
        self.window.set_needs_layout()
        self.Apply_button.enabled = False
        self.Filter_button.is_on = False
        if hasattr(self,'colorbar'):
            self.colorbar.visible = False

    def Target_View_mouse(self, new_val, is_dbl_click):
        self.Target_button_del.enabled = False if self.Model_lib[new_val].type & o3d.io.CONTAINS_TRIANGLES else True
        self.Selected['Target'] = self.Model_lib[new_val]
        self.ICP_button.enabled = self.ICP_button_enabled()
        if hasattr(self,'Coeff') and self.Selected['Target'].type & o3d.io.CONTAINS_TRIANGLES and is_dbl_click:
            self.window.show_dialog(self.Sampling_dialog())
        self.Change_Tbutton.enabled = False if self.Model_lib[new_val].type & o3d.io.CONTAINS_TRIANGLES else True 

    def Data_View_mouse(self, new_val, is_dbl_click):
        self.Data_button_del.enabled = True
        self.Selected['Data'] = self.Model_lib[new_val]
        self.SagErr_cal_button.enabled = True if hasattr(self.Selected['Data'],'Surface') and hasattr(self,'Coeff') else False 
        self.ICP_button.enabled = self.ICP_button_enabled()
        self.Change_Dbutton.enabled = True 
        if is_dbl_click:
            self.active_model = self.Selected['Data']
            self.Data_name.text_value = self.active_model.name
            self.visible_check.checked = self.active_model.visible
            self.tabs.selected_index = 1
            self.window.set_needs_layout()
            self.colorbar.visible = True if hasattr(self.active_model,'SagErr') else False
            self.picked_idx = []
            self.label_list = []
            self.Update_Result(False)

    def Sampling_dialog(self):
        dlg = gui.Dialog('')
        self.Surface = gui.Combobox()
        for col in  self.Coeff.sur:
            self.Surface.add_item(col)
        self.Surface.selected_index = self.Sampling[0]
        self.Xs = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.Ys = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.Xs.double_value = self.Sampling[1]
        self.Ys.double_value = self.Sampling[2]
        Xh = gui.Horiz(0.25 * self.em)
        Yh = gui.Horiz(0.25 * self.em)
        Xh.add_child(gui.Label('X')); Xh.add_child(self.Xs)
        Yh.add_child(gui.Label('Y')); Yh.add_child(self.Ys)
        dig_done = gui.Button('Done')
        dig_done.vertical_padding_em = 0
        dig_done.set_on_clicked(self.Sampling_dialog_done)
        vert = gui.Vert(0, gui.Margins(self.em, self.em, self.em, self.em))
        vert.add_child(gui.Label('Choose Surface'))
        vert.add_child(self.Surface)
        vert.add_fixed(self.em)
        vert.add_child(gui.Label('Sampling rate'))
        vert.add_child(Xh)
        vert.add_child(Yh)
        vert.add_fixed(0.5 * self.em)
        vert.add_child(dig_done)
        dlg.add_child(vert)
        return dlg
    
    def Sampling_dialog_done(self):
        self.Sampling = [self.Surface.selected_index,self.Xs.double_value,self.Ys.double_value]
        self.window.close_dialog()
        S = self.Coeff.sur[self.Sampling[0]]
        try:
            Obj = self.Coeff.Sampling_Surface(self.Model_lib[self.Target_View.selected_value],S,*self.Sampling[1:])
            self.Model_lib[Obj.name] = Obj
            self.Model_lib[Obj.name].visible = True
            if Obj.name not in self.Target_list:
                self.Target_list.append(Obj.name)
                self.Target_View.set_items(self.Target_list)
            self._scene.scene.add_geometry(Obj.name, Obj.cloud, self.material_cloud)
        except Exception:
            print(S,'Edge detection Failed')
        self.window.close_dialog()

    def Master_visible(self):
        self.Option_menu.set_checked(13,not self.Option_menu.is_checked(13))
        self.Model_lib[self.Master_name].visible = self.Option_menu.is_checked(13)
        self.Visible_Control(self.State)

    def Visible_Control(self,State):
        Model_list = list(self.Model_lib)
        if State == 'Main':
            for name in Model_list:
                self._scene.scene.show_geometry(name,self.Model_lib[name].visible)
                if self.Model_lib[name].type & o3d.io.CONTAINS_TRIANGLES:
                    self._scene.scene.show_geometry('wire_'+name,self.Model_lib[name].visible)
            if hasattr(self,'Master_name'):
                bounds = self.Model_lib[self.Master_name].mesh.get_axis_aligned_bounding_box()
                self._scene.center_of_rotation = bounds.get_center()
                #self._scene.setup_camera(60, bounds, bounds.get_center())
            self.Menu.set_enabled(1,True)
            self.Menu.set_enabled(11,True)
            self.Menu.set_enabled(12,True)
            if hasattr(self,'Master_name'):
                self.Menu.set_enabled(13,True)
            else:
                self.Menu.set_enabled(13,False)

        elif State == 'Delete' or State == 'MICP_pickup':
            self._scene.scene.show_geometry(self.active_model.name,True)
            bounds = self.active_model.cloud.get_axis_aligned_bounding_box()
            Model_list.remove(self.active_model.name)
            self.Menu.set_enabled(1,False)
            self.Menu.set_enabled(11,False)
            self.Menu.set_enabled(12,False)
            self.Menu.set_enabled(13,False)
            for name in Model_list:
                self._scene.scene.show_geometry(name,False)
                if self.Model_lib[name].type & o3d.io.CONTAINS_TRIANGLES:
                    self._scene.scene.show_geometry('wire_'+name,False)
            if State == 'Delete':
                self._scene.setup_camera(60, bounds, bounds.get_center())
            else:
                self._scene.center_of_rotation = bounds.get_center()
        
        elif State == 'MICP_Target' or 'MICP_Data':
            for name in [self.Selected['Target'].name,self.Selected['Data'].name,self.Master_name]:
                self._scene.scene.show_geometry(name,True)
                if self.Model_lib[name].type & o3d.io.CONTAINS_TRIANGLES:
                    self._scene.scene.show_geometry('wire_'+name,True)
                Model_list.remove(name)
            for name in Model_list:
                self._scene.scene.show_geometry(name,False)

    def Delete_mode(self):
        if self.State == 'Main':
            self.tabs.selected_index = 2
            self.State = 'Delete'
            self.window.title = self.State
            self.window.set_needs_layout()
            self.Correct_button.set_on_clicked(self.Delete_mode)
            self.picked_idx = []
            self.MouseSelect = []
            self.origin_colors = deepcopy(self.active_model.cloud.colors)
            self.Visible_Control(self.State)
            self.Clean_Mark()
        elif self.State == 'Delete':
            self.State = 'Main'
            self.window.title = self.State
            self.tabs.selected_index = 1
            self.window.set_needs_layout()
            self.active_model.cloud = self.active_model.cloud.select_by_index(self.picked_idx,invert=True)
            if hasattr(self.active_model,'SagErr'):
                self.active_model.SagErr = np.delete(self.active_model.SagErr,self.picked_idx)
                self.Update_Result()
            self.Update_Cloud(self.active_model)
            self.Visible_Control(self.State)
            del self.MouseSelect, self.origin_colors
            self.picked_idx = []

    def Cancel(self):
        if self.State == 'Delete':
            self.State = 'Main'
            self.window.title = self.State
            self.tabs.selected_index = 1
            self.window.set_needs_layout()
            self.picked_idx = []
            self.active_model.cloud.colors = self.origin_colors
            self.Update_Cloud(self.active_model)
            del self.MouseSelect, self.origin_colors
        elif self.State[:4] == 'MICP':
            self.State = 'Main'
            self.window.title = self.State
            self.tabs.selected_index = 0
            self.window.set_needs_layout()
            for i in range(3):
                self._scene.scene.remove_geometry('MICP_Target'+'_sphere'+str(i))
                self._scene.scene.remove_geometry('MICP_Data'+'_sphere'+str(i))
        self.Visible_Control(self.State)

    def Clean_Mark(self):
        for i in range(3):
                self._scene.scene.remove_geometry('MICP_Target'+'_sphere'+str(i))
                self._scene.scene.remove_geometry('MICP_Data'+'_sphere'+str(i))
        if hasattr(self,'label_list'):
            for i,label in enumerate(self.label_list):
                self._scene.remove_3d_label(label)
                self._scene.scene.remove_geometry('label_sphere'+str(i))
            self.label_list = []

    def on_mouse_widget3d(self, event):
        def draw_point():
            if self.State == 'Delete':
                self.active_model.cloud.colors = deepcopy(self.origin_colors)
                np.asarray(self.active_model.cloud.colors)[self.picked_idx] = [0,0,0]
                self.Update_Cloud(self.active_model)
            elif self.State[:4] == 'MICP':
                if len(self.picked_idx) > 3:
                    self.picked_idx = self.picked_idx[:3]
                    print('The number of picked-up points is over 3')
                else:
                    color = ['orange','cyan','magenta','red']
                    for i in range(3):
                        self._scene.scene.remove_geometry(self.State+'_sphere'+str(i))
                    for i,idx in enumerate(self.picked_idx):
                        true_point = np.asarray(self.active_model.cloud.points)[idx]
                        sphere = o3d.geometry.TriangleMesh.create_sphere(0.15).translate(true_point)
                        sphere.paint_uniform_color(ColorConverter.to_rgb(color[i]))
                        self._scene.scene.add_geometry(self.State+'_sphere'+str(i),sphere,self.material_cloud)
            elif self.tabs.selected_index == 1 and hasattr(self.active_model,'SagErr'):
                self.Clean_Mark()
                for i,idx in enumerate(self.picked_idx):
                    true_point = np.asarray(self.active_model.cloud.points)[idx]
                    dz = self.active_model.SagErr[idx]
                    label3d = self._scene.add_3d_label(true_point, '%.3f'%dz)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(0.25).translate(true_point)
                    sphere.paint_uniform_color(ColorConverter.to_rgb('k'))
                    self._scene.scene.add_geometry('label_sphere'+str(i),sphere,self.material_cloud)
                    self.label_list.append(label3d)

        def depth_callback(depth_image):
            if len(self.MouseSelect) == 2:
                start,end = np.min(self.MouseSelect,axis = 0),np.max(self.MouseSelect,axis = 0)
                depth = np.asarray(depth_image)[start[1]:end[1], start[0]:end[0]]
                xyd = [np.append(np.argwhere(depth==d)[0,::-1]+start,d) for d in set(depth.ravel())-set([1])]
            else:
                x,y = self.MouseSelect[0]
                d = np.asarray(depth_image)[y, x]
                xyd = [[x,y,d]] if d != 1 else []
            self.MouseSelect = []
            for x,y,d in xyd:
                world = self._scene.scene.camera.unproject(x,y,d, self._scene.frame.width, self._scene.frame.height)
                idx = self.cacl_prefer_indicate(world)
                if idx in self.picked_idx:
                    self.picked_idx.remove(idx)
                else:
                    self.picked_idx.append(idx)
            gui.Application.instance.post_to_main_thread(self.window, draw_point)

        x = event.x- self._scene.frame.x
        y = event.y- self._scene.frame.y

        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT) and event.is_modifier_down(gui.KeyModifier.CTRL)\
            and (self.tabs.selected_index == 2 or (self.tabs.selected_index == 1 and hasattr(self.active_model,'SagErr'))):
            self._scene.set_view_controls(gui.SceneWidget.Controls.PICK_POINTS)
            self.MouseSelect = [[x,y]]
            self._scene.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED

        elif event.type == gui.MouseEvent.Type.DRAG and event.is_button_down(gui.MouseButton.RIGHT) and event.is_modifier_down(gui.KeyModifier.CTRL) \
            and self.tabs.selected_index == 2 and self.State == 'Delete':
            self._scene.set_view_controls(gui.SceneWidget.Controls.PICK_POINTS)
            if len(self.MouseSelect) > 1:
                self.MouseSelect[-1] = [x,y]
                start = np.min(self.MouseSelect,axis = 0)+[self._scene.frame.x,self._scene.frame.y]
                end = np.max(self.MouseSelect,axis = 0)+[self._scene.frame.x,self._scene.frame.y]
                L,W = abs(start[0]-end[0]),abs(start[1]-end[1])
                if L>self.em and W>self.em:
                    self._RectSelect.visible = True
                    self._RectSelect.frame = gui.Rect(start[0], start[1], L, W)
                else:
                    self._RectSelect.visible = False
            else:
                self.MouseSelect += [[x,y]]
                self._RectSelect.visible = False
            return gui.Widget.EventCallbackResult.HANDLED

        elif event.type == gui.MouseEvent.Type.BUTTON_UP \
            and self.tabs.selected_index == 2 and self.State == 'Delete':
            self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
            if self._RectSelect.visible:
                self._scene.scene.scene.render_to_depth_image(depth_callback)
                self._RectSelect.visible = False
            else:
                self.RectSelect = []
            return gui.Widget.EventCallbackResult.HANDLED
        else:
            return gui.Widget.EventCallbackResult.IGNORED
    
    def cacl_prefer_indicate(self, point):
        pcd_tree = o3d.geometry.KDTreeFlann(self.active_model.cloud)
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        return idx[0]

    def on_key_widget3d(self, event):
        if self.State[:4] == 'MICP':
            name ='Data' if self.State[5:]=='Target' else 'Target'
            if event.key == gui.KeyName.LEFT_CONTROL and event.type == gui.KeyEvent.DOWN:
                self.Visible_Control('MICP_pickup')
                for name in ['MICP_'+name+'_sphere'+str(i) for i in range(3)]:
                    self._scene.scene.show_geometry(name,False)
            elif event.key == gui.KeyName.LEFT_CONTROL and event.type == gui.KeyEvent.UP:
                self.Visible_Control(self.State)
                for name in ['MICP_Target'+'_sphere'+str(i) for i in range(3)]+['MICP_Data'+'_sphere'+str(i) for i in range(3)]:
                    self._scene.scene.show_geometry(name,True)
            return gui.Widget.EventCallbackResult.HANDLED

        elif self.tabs.selected_index == 1 and event.key == gui.KeyName.LEFT_CONTROL and hasattr(self.active_model,'SagErr'):
            if  event.type == gui.KeyEvent.DOWN :
                self.Back_button.enabled = False
                self.Apply_button.enabled = False
                self.Save_button.enabled = False
                self.window.set_needs_layout()
                self._scene.set_view_controls(gui.SceneWidget.Controls.PICK_POINTS)
                self.Visible_Control('MICP_pickup')

            elif event.type == gui.KeyEvent.UP:
                self.Back_button.enabled = True
                self.Apply_enabled()
                self.Save_button.enabled = True
                self.window.set_needs_layout()
                self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
                self.Visible_Control(self.State)
            return gui.Widget.EventCallbackResult.HANDLED
        else:
            return gui.Widget.EventCallbackResult.IGNORED
    
    def Update_Cloud(self,Model):
        self._scene.scene.remove_geometry(Model.name)
        self._scene.scene.add_geometry(Model.name,Model.cloud,self.material_cloud)

    def Draw_dialog(self):
        def ChangeClass(newitem,newidx):
            Cmaplist.clear_items()
            for col in cmaps_dir[newitem]:
                Cmaplist.add_item(col)
            Cmaplist.selected_index = 0
            Check_R.checked = False
        def Setting_dialog_done():
            self.ColorMap = [CmapClass.selected_text,Cmaplist.selected_text,'_r' if Check_R.checked else '',
                             Fix_Check.checked,Range_min.double_value,Range_max.double_value]
            Ori_Active_name = deepcopy(self.active_model.name) if hasattr(self,'active_model') else ''
            for name in  list(self.Model_lib):
                if self.Model_lib[name].type & o3d.io.CONTAINS_POINTS:
                    self.active_model = self.Model_lib[name]
                    if hasattr(self.active_model,'SagErr'):
                        self.Update_Result()
                        self.Update_Cloud(self.active_model)
            if Ori_Active_name:
                self.active_model = self.Model_lib[Ori_Active_name]
            self.window.close_dialog()
            self.Visible_Control(self.State)

        dlg = gui.Dialog('')
        CmapClass = gui.Combobox()
        for col in  cmaps_dir:
            CmapClass.add_item(col)
        CmapClass.selected_text = self.ColorMap[0]
        CmapClass.set_on_selection_changed(ChangeClass)
        Cmaplist = gui.Combobox()
        for col in  cmaps_dir[CmapClass.selected_text]:
            Cmaplist.add_item(col)
        Cmaplist.selected_text = self.ColorMap[1]
        Check_R = gui.Checkbox('R')
        Check_R.checked = True if self.ColorMap[2] == '_r' else False

        dig_done = gui.Button('Done')
        dig_done.vertical_padding_em = 0
        dig_done.set_on_clicked(Setting_dialog_done)
        dig_cancel = gui.Button('Cancel')
        dig_cancel.vertical_padding_em = 0
        dig_cancel.set_on_clicked(self.window.close_dialog)
        
        buttonbar = gui.Horiz(0.25 * self.em)
        buttonbar.add_stretch()
        buttonbar.add_child(dig_cancel)
        buttonbar.add_fixed(0.2*self.em)
        buttonbar.add_child(dig_done)

        Fix_Check = gui.Checkbox('')
        Fix_Check.checked = self.ColorMap[3]
        Range_min = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        Range_min.double_value = self.ColorMap[4]
        Range_max = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        Range_max.double_value = self.ColorMap[5]
        Max_h = gui.Horiz(0 * self.em)
        Max_h.add_child(gui.Label('Min'))
        Max_h.add_child(Range_min)
        Min_h = gui.Horiz(0 * self.em)
        Min_h.add_child(gui.Label('Max'))
        Min_h.add_child(Range_max)
        
        vgrid = gui.VGrid(4, 0.2*self.em,gui.Margins(0.5*self.em, 0.2*self.em, 0.2*self.em, 0))
        vgrid.add_child(gui.Label('Colormap'))
        vgrid.add_child(CmapClass)
        vgrid.add_child(Cmaplist)
        vgrid.add_child(Check_R)
        vgrid.add_child(gui.Label('Range'))
        vgrid.add_child(Max_h)
        vgrid.add_child(Min_h)
        vgrid.add_child(Fix_Check)
        

        vert = gui.Vert(0, gui.Margins(self.em, self.em, self.em, self.em))
        vert.add_child(gui.Label('Draw Setting'))
        vert.add_fixed(0.2*self.em)
        vert.add_child(vgrid)
        vert.add_fixed(0.2*self.em)
        vert.add_fixed(0.5*self.em)
        vert.add_child(buttonbar)
        dlg.add_child(vert)
        self.window.show_dialog(dlg)

    def ICP_dialog(self):
        def ChangeWeight(newitem,newidx):
            if np.isnan(k_value.double_value):
                k_value.double_value = 0.05
            if newitem == 'None' or newitem == 'L1Loss' or newitem == 'L2Loss':
                k_value.enabled = False
                k_value.double_value = np.nan
            else:
                k_value.enabled = True

        def ChangeICP(newitem,newidx):
            if np.isnan(k_value.double_value):
                k_value.double_value = 0.05
            if newitem == 'PointToPoint':
                WeightClass.selected_text = 'None'
                WeightClass.enabled = False
                k_value.enabled = False
                k_value.double_value = np.nan
            else:
                k_value.enabled = False if WeightClass.selected_text == 'None' else True
                WeightClass.enabled = True

        def ICP_dialog_done():
            self.ICP_parameter = [Manual_check.checked,ICP_select.selected_text,Distance_threshold.double_value,WeightClass.selected_text,k_value.double_value]
            if self.ICP_parameter[0]==False:
                self.ICP_button.set_on_clicked(self.ICP_Algorithm)
                self.ICP_button.text = 'Direct ICP'
            else:
                self.ICP_button.set_on_clicked(self.Manual_ICP)
                self.ICP_button.text = 'Manual ICP'
            self.window.close_dialog()
        
        dlg = gui.Dialog('')
        dig_done = gui.Button('Done')
        dig_done.vertical_padding_em = 0
        dig_done.set_on_clicked(ICP_dialog_done)
        dig_cancel = gui.Button('Cancel')
        dig_cancel.vertical_padding_em = 0
        dig_cancel.set_on_clicked(self.window.close_dialog)
        Manual_check = gui.Checkbox('Manual')
        Manual_check.checked = self.ICP_parameter[0]
        
        buttonbar = gui.Horiz(0.25 * self.em)
        buttonbar.add_child(Manual_check)
        buttonbar.add_stretch()
        buttonbar.add_child(dig_cancel)
        buttonbar.add_fixed(0.2*self.em)
        buttonbar.add_child(dig_done)

        ICP_select = gui.Combobox()
        for item in  ICP_Class:
            ICP_select.add_item(item)
        ICP_select.selected_text = self.ICP_parameter[1]
        ICP_select.set_on_selection_changed(ChangeICP)
        Distance_threshold = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        Distance_threshold.double_value = self.ICP_parameter[2]
        WeightClass = gui.Combobox()
        for item in Weight_Function:
            WeightClass.add_item(item)
        WeightClass.selected_text = self.ICP_parameter[3]
        WeightClass.set_on_selection_changed(ChangeWeight)
        WeightClass.enabled = False if self.ICP_parameter[1] == 'PointToPoint' else True
        k_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        k_value.double_value = self.ICP_parameter[4]
        k_value.enabled = False if np.isnan(k_value.double_value) or WeightClass.selected_text == 'None' else True

        vgrid = gui.VGrid(2, 0.1*self.em,gui.Margins(0.5*self.em, 0.2*self.em, 0.2*self.em, 0))
        vgrid.add_child(gui.Label('ICP Algorithm'))
        vgrid.add_child(ICP_select)
        vgrid.add_child(gui.Label('Distance threshold'))
        vgrid.add_child(Distance_threshold)
        vgrid.add_child(gui.Label('Weight Function'))
        vgrid.add_child(WeightClass)
        vgrid.add_child(gui.Label('k value'))
        vgrid.add_child(k_value)

        vert = gui.Vert(0, gui.Margins(self.em, self.em, self.em, self.em))
        vert.add_child(gui.Label('Robust Kernel'))
        vert.add_fixed(0.2*self.em)
        vert.add_child(vgrid)
        vert.add_fixed(0.2*self.em)
        vert.add_child(buttonbar)
        dlg.add_child(vert)
        self.window.show_dialog(dlg)

    def Set_ICP_Algorithm(self):
        ICP_Algorithm = ICP_Class[self.ICP_parameter[1]]
        if self.ICP_parameter[3] != 'None':
            loss = Weight_Function[self.ICP_parameter[3]]
            if hasattr(Weight_Function[self.ICP_parameter[3]],'k'):
                loss.k = self.ICP_parameter[4]
            ICP_Algorithm.kernel = loss
        return ICP_Algorithm

    def Manual_ICP(self):
        if self.State == 'Main':
            self.tabs.selected_index = 2
            self.State = 'MICP_Target'
            self.Visible_Control(self.State)
            self.window.title = self.State
            self.window.set_needs_layout()
            self.Correct_button.set_on_clicked(self.Manual_ICP)
            self.picked_idx = []
            self.active_model = self.Selected['Target']
            bounds = self.active_model.cloud.get_axis_aligned_bounding_box()
            self._scene.setup_camera(60, bounds, bounds.get_center())
        
        elif self.State == 'MICP_Target' and len(self.picked_idx) >= 3:
            self.Target_pickup = deepcopy(self.picked_idx)
            self.State = 'MICP_Data'
            self.window.title = self.State
            self.picked_idx = []
            self.active_model = self.Selected['Data']
            bounds = self.active_model.cloud.get_axis_aligned_bounding_box()
            self._scene.setup_camera(60, bounds, bounds.get_center())
            
        elif self.State == 'MICP_Data' and len(self.picked_idx) >= 3:
            self.State = 'Main'
            self.window.title = self.State
            self.tabs.selected_index = 0
            self.window.set_needs_layout()
            self.Data_pickup = deepcopy(self.picked_idx)
            self.Clean_Mark()
            
            self.ICP_Algorithm()
            del self.picked_idx,self.active_model

    def ICP_Algorithm(self):
        source = self.Selected['Data'].cloud
        target = self.Selected['Target'].cloud
        if hasattr(self,'Data_pickup') and hasattr(self,'Target_pickup'):
            corr = np.vstack((self.Data_pickup,self.Target_pickup)).T
            p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
            trans_init = p2p.compute_transformation(source, target,o3d.utility.Vector2iVector(corr))
            del self.Data_pickup, self.Target_pickup
        else:
            trans_init = np.identity(4)
        ICP_Algorithm = self.Set_ICP_Algorithm()
        max_correspondence_distance = self.ICP_parameter[2]
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 3000)
        reg = o3d.pipelines.registration.registration_icp(source, target, max_correspondence_distance, trans_init, ICP_Algorithm, criteria)
        TranMatrix = reg.transformation
        source.transform(TranMatrix)

        self.Selected['Data'].Surface = self.Selected['Target'].Surface
        #self._scene.scene.remove_geometry(self.Selected['Target'].name)
        self.Update_Cloud(self.Selected['Data'])
        self.Visible_Control(self.State)

        #self.Target_list.remove(self.Selected['Target'].name)
        #self.Target_View.set_items(self.Target_list)
        #del self.Model_lib[self.Selected['Target'].name], self.Selected['Target'],
        self.SagErr_cal_button.enabled = True
        self.ICP_button.enabled = False

    def SagErr_cal(self):
        self.active_model = self.Model_lib[self.Data_View.selected_value]
        S = self.active_model.Surface
        pcd = deepcopy(self.active_model.cloud)
        pcd.transform(np.linalg.inv(self.Coeff.Matrix44(S)))
        xyz = np.asarray(pcd.points)
        zt = self.Coeff.Sag_Z(*xyz[:,:2].T,S)
        self.active_model.SagErr = (-1)**self.active_model.z_direction*1000*(xyz[:,2]-zt)
        self.Update_Result()
        self.Update_Cloud(self.active_model)
        self.SagErr_cal_button.enabled = False
        self.Filter_button.enabled = True
    
    def Update_Result(self, calculate = True):
        self.Direction.selected_index = self.active_model.z_direction
        self.Scale_value.set_value(self.active_model.scale)
        if hasattr(self.active_model,'SagErr'):
            std = np.std(self.active_model.SagErr)
            vmax = self.active_model.SagErr.max()
            vmin = self.active_model.SagErr.min()
            self.Max_value.set_value(vmax)
            self.Min_value.set_value(vmin)
            self.FilterMax_value.set_value(vmax)
            self.FilterMin_value.set_value(vmin)
            self.Avg_value.set_value(np.average(self.active_model.SagErr))
            self.Std_value.set_value(std)
            if calculate:
                VMin =self.ColorMap[4] if self.ColorMap[3] else -3*std
                VMax =self.ColorMap[5] if self.ColorMap[3] else 3*std
                fig_colorbar, mapping = self.SagErr_colorbar(VMin,VMax,self.ColorMap[1]+self.ColorMap[2])
                fig_histogram = self.SagErr_histogram(self.active_model.SagErr)
                self.active_model.colorbar = fig_colorbar
                self.active_model.histogram = fig_histogram
                self.active_model.cloud.colors = o3d.utility.Vector3dVector(mapping.to_rgba(self.active_model.SagErr)[:,:3])
            self.colorbar.update_image(self.active_model.colorbar)
            self.histogram.update_image(self.active_model.histogram)
        else:
            self.Max_value.set_value(np.nan)
            self.Min_value.set_value(np.nan)
            self.FilterMax_value.set_value(np.nan)
            self.FilterMin_value.set_value(np.nan)
            self.Avg_value.set_value(np.nan)
            self.Std_value.set_value(np.nan)
            self.histogram.update_image(o3d.geometry.Image(np.zeros((10*self.em,16*self.em,3),dtype=np.uint8)))

    def SagErr_colorbar(self,vmin,vmax,cmap):
        fig, ax = plt.subplots(figsize = (4,0.8))
        fig.subplots_adjust(left=0.04, bottom=0.6, right = 0.96, top = 0.98)
        digi = 0 if int(f"{vmax:.0E}"[2:])>0 else int(f"{vmax:.0E}"[2:])-1
        dv = 10**digi
        norm = Normalize(vmin = vmin-dv, vmax = vmax+dv)
        mapping = ScalarMappable(norm=norm, cmap = cmap)
        fig.colorbar(mapping,cax = ax, orientation = 'horizontal',ticks=np.round(np.linspace(vmin,vmax,7),abs(digi)))
        fig.set_facecolor([235/256,234/256,234/256])
        ax.set_xlabel(r'$\mu$m', loc='right')
        fig.canvas.draw()
        fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig = o3d.geometry.Image(fig)
        plt.close('all')
        return fig,mapping

    def SagErr_histogram(self,SagErr):
        fig, ax = plt.subplots(figsize = (2.7,2))
        fig.subplots_adjust(left=0.01, bottom=0.25, right = 0.99, top = 0.98)
        Yhis,Xhis = np.histogram(SagErr, 25)
        ax.bar(Xhis[:-1],Yhis,width=Xhis[1]-Xhis[0],color='w',edgecolor = 'k')
        ax.set_xlabel('Sag Error ('+r'$\mu$m'+')',color = 'w',fontsize = 'small')
        ax.tick_params(axis='x',colors='w',labelsize= 'small')
        ax.get_yaxis().set_visible(False)
        ax.set_facecolor([26/256,26/256,26/256])
        fig.set_facecolor([45/256,45/256,45/256])
        fig.canvas.draw()
        fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig = o3d.geometry.Image(fig)
        plt.close('all')
        return fig
    
    def Apply_enabled(self, new_val = 0, new_idx = 0):
        if self.FilterMax_value.double_value <= self.FilterMin_value.double_value:
            print('Number Error')
            self.FilterMax_value.set_value(self.Max_value.double_value)
            self.FilterMin_value.set_value(self.Min_value.double_value)
        elif self.Direction.selected_index != self.active_model.z_direction or self.Filter_button.is_on:
            self.Apply_button.enabled = True
        else:
            self.Apply_button.enabled = False

    def Change_visible(self,checkstate):
        self._scene.scene.show_geometry(self.active_model.name,checkstate)
        self.active_model.visible = checkstate

    def Apply_clicked(self):
        self.Clean_Mark()
        if self.Filter_button.is_on and hasattr(self.active_model,'SagErr'):
            idx = np.argwhere((self.FilterMin_value.double_value<=self.active_model.SagErr)*(self.active_model.SagErr<=self.FilterMax_value.double_value))[:,0]
            if len(idx)<2:
                print('Filter Error')
            elif len(idx) != len(self.active_model.cloud.points):
                self.active_model.cloud = self.active_model.cloud.select_by_index(idx)
                self.active_model.SagErr = deepcopy(self.active_model.SagErr[idx])
                self.active_model.cloud.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 1, max_nn = 25))
                self.active_model.cloud.estimate_covariances(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 1, max_nn = 25))
                self.active_model.cloud.normalize_normals()
        if self.active_model.z_direction != self.Direction.selected_index:
            self.active_model.z_direction = deepcopy(self.Direction.selected_index)
            if hasattr(self.active_model,'SagErr'):
                self.active_model.SagErr *= -1
        self.Update_Result()
        self.Update_Cloud(self.active_model)
        self.Apply_button.enabled = False
        self.Filter_button.is_on = False

    def Save_clicked(self):
        pass

            

#%%
if __name__ == '__main__':
    gui.Application.instance.initialize()
    app = AppWindow(1024, 768)
    gui.Application.instance.run()
# %%