from copy import deepcopy
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import Model

class AppWindow:
    Model_lib = {}
    Data_list = set([])
    Target_list = set([])
    Selected = {}

    #Default_value
    State = 'Main'
    File_Load = [None,1]
    Sampling = [0, 0.5, 0.5]
    ColorMap = ['Miscellaneous','gist_rainbow','_r']
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

        self._info = gui.Label('')
        self._info.visible = False
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
        #self.Change_Tbutton.set_on_clicked(self.Change_Target)
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
        #self.Change_Dbutton.set_on_clicked(self.Change_Data)
        self.Change_Dbutton.enabled = False

        self.ICP_button = gui.Button('Manual ICP')
        self.ICP_button.horizontal_padding_em = 1.5
        self.ICP_button.vertical_padding_em = 1
        #self.ICP_button.set_on_clicked(self.Manual_ICP)
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
        #self.SagErr_cal_button.set_on_clicked(self.SagErr_cal)

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
        #self.Direction.set_on_selection_changed(self.Apply_enabled)
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
        #self.Filter_button.set_on_clicked(self.Apply_enabled)

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
        #self.Apply_button.set_on_clicked(self.Apply_clicked)

        self.Back_button = gui.Button('Back')
        self.Back_button.enabled = True
        self.Back_button.horizontal_padding_em = 0
        self.Back_button.vertical_padding_em = 0
        self.Back_button.set_on_clicked(self.Back_Main)

        self.Save_button = gui.Button('Save')
        self.Save_button.enabled = True
        self.Save_button.horizontal_padding_em = 0
        self.Save_button.vertical_padding_em = 0
        #self.Save_button.set_on_clicked(self.Save_clicked)

        self.visible_check = gui.Checkbox('')
        #self.visible_check.set_on_checked(self.Change_visible)
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
        #self.Cancel_button.set_on_clicked(self.Cancel)

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
        #self._scene.set_on_mouse(self._on_mouse_widget3d)
        #self._scene.set_on_key(self._on_key_widget3d)
        self.window.add_child(self._scene)

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.tabs)
        self.window.add_child(self._info)
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
        Option_menu = gui.Menu()
        Option_menu.add_item('Drawing settings',11)
        Option_menu.add_item('ICP settings',12)
        self.Menu.add_menu("File", file_menu)
        self.Menu.add_menu("Option", Option_menu)
        gui.Application.instance.menubar = self.Menu
        #self.window.set_on_menu_item_activated(1, self.csv2xyz_dialog)
        #self.window.set_on_menu_item_activated(11, self.Draw_dialog)
        #self.window.set_on_menu_item_activated(12, self.ICP_dialog)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        visible_panel = self.Panel[self.tabs.selected_index]
        height = min(r.height,visible_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)
        self.tabs.frame = gui.Rect(r.get_right() - width, r.y, width,height)
        
        pref = self._info.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self._info.frame = gui.Rect(r.x, r.get_bottom()-pref.height, pref.width, pref.height)
        pref = self.Manual.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self.Manual.frame = gui.Rect(r.get_right()-pref.width, r.get_bottom()-pref.height, pref.width, pref.height)
        pref = self.colorbar.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self.colorbar.frame = gui.Rect(r.get_right()-width-pref.width, r.y, pref.width, pref.height)

        self.window.close_dialog()

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
        self.Model_lib[Geometry.name] = Geometry
        if Geometry.type & o3d.io.CONTAINS_TRIANGLES:
            self._scene.scene.add_geometry(Geometry.name, Geometry.mesh, self.material_mesh)
            self._scene.scene.add_geometry('wire_'+Geometry.name, Geometry.wire, self.material_wire)
            bounds = Geometry.mesh.get_axis_aligned_bounding_box()
            self.Target_list.add(Geometry.name)
            self.Target_View.set_items(list(self.Target_list))
        elif Geometry.type & o3d.io.CONTAINS_POINTS:
            Geometry.z_direction = 0
            self._scene.scene.add_geometry(Geometry.name, Geometry.cloud, self.material_cloud)
            bounds = Geometry.cloud.get_axis_aligned_bounding_box()
            self.Data_list.add(Geometry.name)
            self.Data_View.set_items(list(self.Data_list))
        else:
            bounds = o3d.geometry.AxisAlignedBoundingBox(np.array([-1,-1,-1]),np.array([1,1,1]))
        self._scene.setup_camera(60, bounds, bounds.get_center())

    def Target_Delete(self):
        del self.Model_lib[self.Target_View.selected_value], self.Selected['Target']
        self._scene.scene.remove_geometry(self.Target_View.selected_value)
        self._scene.scene.remove_geometry('wire_'+self.Target_View.selected_value)
        self.Target_list.remove(self.Target_View.selected_value)
        self.Target_View.set_items(list(self.Target_list))
        self.Target_button_del.enabled = False
        self.ICP_button.enabled = False

    def Data_Delete(self):
        del self.Model_lib[self.Data_View.selected_value], self.Selected['Data']
        self._scene.scene.remove_geometry(self.Data_View.selected_value)
        self.Data_list.remove(self.Data_View.selected_value)
        self.Data_View.set_items(list(self.Data_list))
        self.Data_button_del.enabled = False
        self.ICP_button.enabled = False
        
    def ICP_button_enabled(self):
        if 'Target' in self.Selected and self.Selected['Target'].type & o3d.io.CONTAINS_POINTS and 'Data' in self.Selected:
            return True
        else:
            return False

    def Back_Main(self):
        del self.active_model
        self.tabs.selected_index = 0
        self.window.set_needs_layout()
        self.Apply_button.enabled = False
        if hasattr(self,'colorbar'):
            self.colorbar.visible = False

    def Target_View_mouse(self, new_val, is_dbl_click):
        self.Target_button_del.enabled = True
        self.Selected['Target'] = self.Model_lib[new_val]
        self.ICP_button.enabled = self.ICP_button_enabled()
        if hasattr(self,'Coeff') and self.Selected['Target'].type & o3d.io.CONTAINS_TRIANGLES and is_dbl_click:
            self.window.show_dialog(self.Sampling_dialog())

    def Data_View_mouse(self, new_val, is_dbl_click):
        self.Data_button_del.enabled = True
        self.Selected['Data'] = self.Model_lib[new_val]
        self.ICP_button.enabled = self.ICP_button_enabled()
        if is_dbl_click:
            self.active_model = self.Selected['Data']
            self.Data_name.text_value = self.active_model.name
            self.visible_check.checked = self.active_model.visible
            self.tabs.selected_index = 1
            self.window.set_needs_layout()
            self.colorbar.visible = True if hasattr(self.active_model,'SagErr') else False

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
            self.Target_list.add(Obj.name)
            self.Target_View.set_items(list(self.Target_list))
            self._scene.scene.add_geometry(Obj.name, Obj.cloud, self.material_cloud)
        except Exception:
            print(S,'Edge detection Failed')
        self.window.close_dialog()

    def Visible_control(self,State):
        if State == 'Main':
            Model_list = list(self.Model_lib)
            for name in Model_list:
                self._scene.scene.show_geometry(name,True)
                if self.Model_lib[name].type & o3d.io.CONTAINS_TRIANGLES:
                    self._scene.scene.show_geometry('wire_'+name,True)
        elif State == 'Delete':
            self._scene.scene.show_geometry(self.active_model.name,True)
            Model_list = list(self.Model_lib)
            Model_list.remove(self.active_model.name)
            for name in Model_list:
                self._scene.scene.show_geometry(name,False)
                if self.Model_lib[name].type & o3d.io.CONTAINS_TRIANGLES:
                    self._scene.scene.show_geometry('wire_'+name,False)
            bounds = self.active_model.cloud.get_axis_aligned_bounding_box()
            self._scene.setup_camera(60, bounds, bounds.get_center())

    def Delete_mode(self):
        if self.State == 'Main':
            self.tabs.selected_index = 2
            self.State = 'Delete'
            self.window.title = self.State
            self.window.show_menu(False)
            self.window.set_needs_layout()
            self.Correct_button.set_on_clicked(self.Delete_mode)
            self.picked_idx = []
            self.RectSelect_idx = []
            self.origin_colors = deepcopy(self.active_model.cloud.colors)
            self.Visible_control(self.State)
        elif self.State == 'Delete':
            self.State = 'Main'
            self.window.title = self.State
            self.tabs.selected_index = 1
            self.window.show_menu(True)
            self.window.set_needs_layout()
            self.Visible_control(self.State)
            

#%%
if __name__ == '__main__':
    gui.Application.instance.initialize()
    app = AppWindow(1024, 768)
    gui.Application.instance.run()
# %%